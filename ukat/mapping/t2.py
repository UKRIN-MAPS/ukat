import os
import nibabel as nib
import numpy as np
import concurrent.futures
from tqdm import tqdm
from scipy.optimize import curve_fit


class T2:
    """
    Attributes
    ----------
    t2_map : np.ndarray
        The estimated T2 values in ms
    t2_err : np.ndarray
        The certainty in the fit of `t2` in ms
    m0_map : np.ndarray
        The estimated M0 values
    m0_err : np.ndarray
        The certainty in the fit of `m0`
    shape : tuple
        The shape of the T2 map
    n_te : int
        The number of TE used to calculate the map
    n_vox : int
        The number of voxels in the map i.e. the product of all dimensions
        apart from TE
    """

    def __init__(self, pixel_array, echo_list, affine, mask=None,
                 noise_threshold=0, method='2p_exp', multithread='auto'):
        """Initialise a T2 class instance.

        Parameters
        ----------
        pixel_array : np.ndarray
            An array containing the signal from each voxel at each echo
            time with the last dimension being time i.e. the array needed to
            generate a 3D T2 map would have dimensions [x, y, z, TE].
        echo_list : list()
            An array of the echo times used for the last dimension of the
            raw data. In milliseconds.
        affine : np.ndarray
            A matrix giving the relationship between voxel coordinates and
            world coordinates.
        mask : np.ndarray, optional
            A boolean mask of the voxels to fit. Should be the shape of the
            desired T2 map rather than the raw data i.e. omit the time
            dimension.
        noise_threshold : float, optional
            Default 0
            Voxels with magnitude less than this threshold will not be used
            when fitting. This can be useful if the noise floor of the data
            is known.
        method : {'2p_exp', '3p_exp'}, optional
            Default `2p_exp`
            The model the data is fit to. 2p_exp uses a two parameter
            exponential model (S = S0 * exp(-t / T2)) whereas 3p_exp uses a
            three parameter exponential model (S = S0 * exp(-t / T2) + b) to
            fit for noise/very long T2 components of the signal.
        multithread : bool or 'auto', optional
            Default 'auto'.
            If True, fitting will be distributed over all cores available on
            the node. If False, fitting will be carried out on a single thread.
            'auto' attempts to apply multithreading where appropriate based
            on the number of voxels being fit.
        """
        # Some sanity checks
        assert (pixel_array.shape[-1]
                == len(echo_list)), 'Number of echoes does not match the ' \
                                    'number of time frames on the last axis ' \
                                    'of pixel_array'
        assert multithread is True \
            or multithread is False \
            or multithread == 'auto', 'multithreaded must be True, ' \
                                      'False or auto. You entered {}' \
            .format(multithread)
        if method != '2p_exp' and method != '3p_exp':
            raise ValueError('method can be 2p_exp or 3p_exp only. You '
                             'specified {}'.format(method))

        self.pixel_array = pixel_array
        self.shape = pixel_array.shape[:-1]
        self.n_te = pixel_array.shape[-1]
        self.n_vox = np.prod(self.shape)
        self.affine = affine
        # Generate a mask if there isn't one specified
        if mask is None:
            self.mask = np.ones(self.shape, dtype=bool)
        else:
            self.mask = mask
            # Don't process any nan values
        self.mask[np.isnan(np.sum(pixel_array, axis=-1))] = False
        self.noise_threshold = noise_threshold
        self.method = method
        self.echo_list = echo_list
        # Auto multithreading conditions
        if multithread == 'auto':
            if self.n_vox > 20:
                multithread = True
            else:
                multithread = False
        self.multithread = multithread

        # Fit data
        if self.method == '2p_exp':
            self.t2_map, self.t2_err, \
                self.m0_map, self.m0_err \
                = self.__fit__()
        elif self.method == '3p_exp':
            self.t2_map, self.t2_err, \
                self.m0_map, self.m0_err, \
                self.b_map, self.b_err \
                = self.__fit__()

    def __fit__(self):

        # Initialise maps
        t2_map = np.zeros(self.n_vox)
        t2_err = np.zeros(self.n_vox)
        m0_map = np.zeros(self.n_vox)
        m0_err = np.zeros(self.n_vox)
        b_map = np.zeros(self.n_vox)
        b_err = np.zeros(self.n_vox)
        mask = self.mask.flatten()
        signal = self.pixel_array.reshape(-1, self.n_te)
        # Get indices of voxels to process
        idx = np.argwhere(mask).squeeze()

        # Multithreaded method
        if self.multithread:
            with concurrent.futures.ProcessPoolExecutor() as pool:
                with tqdm(total=idx.size) as progress:
                    futures = []

                    for ind in idx:
                        signal_thresh = signal[ind, :][
                            signal[ind, :] > self.noise_threshold]
                        echo_list_thresh = self.echo_list[
                            signal[ind, :] > self.noise_threshold]
                        future = pool.submit(self.__fit_signal__,
                                             signal_thresh,
                                             echo_list_thresh)
                        future.add_done_callback(lambda p: progress.update())
                        futures.append(future)

                    results = []
                    for future in futures:
                        result = future.result()
                        results.append(result)

            if self.method == '2p_exp':
                t2_map[idx], t2_err[idx], m0_map[idx], m0_err[idx] = [np.array(
                    row) for row in zip(*results)]
            elif self.method == '3p_exp':
                t2_map[idx], t2_err[idx], \
                    m0_map[idx], m0_err[idx], \
                    b_map[idx], b_err[idx] = \
                    [np.array(row) for row in zip(*results)]

        # Single threaded method
        else:
            with tqdm(total=idx.size) as progress:
                for ind in idx:
                    signal_thresh = signal[ind, :][
                        signal[ind, :] > self.noise_threshold]
                    echo_list_thresh = self.echo_list[
                        signal[ind, :] > self.noise_threshold]
                    if self.method == '2p_exp':
                        t2_map[ind], t2_err[ind], \
                            m0_map[ind], m0_err[ind] \
                            = self.__fit_signal__(signal_thresh,
                                                  echo_list_thresh)
                    elif self.method == '3p_exp':
                        t2_map[ind], t2_err[ind], \
                            m0_map[ind], m0_err[ind], \
                            b_map[ind], b_err[ind] \
                            = self.__fit_signal__(signal_thresh,
                                                  echo_list_thresh)
                    progress.update(1)

        # Reshape results to raw data shape
        t2_map = t2_map.reshape(self.shape)
        t2_err = t2_err.reshape(self.shape)
        m0_map = m0_map.reshape(self.shape)
        m0_err = m0_err.reshape(self.shape)

        if self.method == '2p_exp':
            return t2_map, t2_err, m0_map, m0_err
        elif self.method == '3p_exp':
            b_map = b_map.reshape(self.shape)
            b_err = b_err.reshape(self.shape)
            return t2_map, t2_err, m0_map, m0_err, b_map, b_err

    def __fit_signal__(self, sig, te):

        # Initialise parameters
        if self.method == '2p_exp':
            eq = two_param_eq
            bounds = ([0, 0], [1000, 100000000])
            initial_guess = [20, 10000]
        elif self.method == '3p_exp':
            eq = three_param_eq
            bounds = ([0, 0, 0], [1000, 100000000, 1000000])
            initial_guess = [20, 10000, 500]

        # Fit data to equation
        try:
            popt, pcov = curve_fit(eq, te, sig, p0=initial_guess,
                                   bounds=bounds)
        except (RuntimeError, ValueError):
            popt = np.zeros(3)
            pcov = np.zeros((3, 3))

        # Extract fits and errors from result variables
        if self.method == '2p_exp':
            if popt[0] < bounds[1][0] - 1:
                t2 = popt[0]
                m0 = popt[1]
                err = np.sqrt(np.diag(pcov))
                t2_err = err[0]
                m0_err = err[1]
            else:
                t2, m0, t2_err, m0_err = 0, 0, 0, 0

            return t2, t2_err, m0, m0_err

        elif self.method == '3p_exp':
            if popt[0] < bounds[1][0] - 1:
                t2 = popt[0]
                m0 = popt[1]
                b = popt[2]
                err = np.sqrt(np.diag(pcov))
                t2_err = err[0]
                m0_err = err[1]
                b_err = err[2]
            else:
                t2, m0, t2_err, m0_err, b, b_err = 0, 0, 0, 0, 0, 0

            return t2, t2_err, m0, m0_err, b, b_err

    def r2_map(self):
        """
        Generates the R2 map from the T2 map output by initialising this
        class.

        Parameters
        ----------
        See class attributes in __init__

        Returns
        -------
        r2 : np.ndarray
            An array containing the R2 map generated
            by the function with R2 measured in ms.
        """
        return np.reciprocal(self.t2_map)

    def to_nifti(self, output_directory=os.getcwd(), base_file_name='Output',
                 maps='all'):
        """Exports some of the T2 class attributes to NIFTI.
                        
        Parameters
        ----------
        output_directory : string, optional
            Path to the folder where the NIFTI files will be saved.
        base_file_name : string, optional
            Filename of the resulting NIFTI. This code appends the extension.
            Eg., base_file_name = 'Output' will result in 'Output.nii.gz'.
        maps : list or 'all', optional
            List of maps to save to NIFTI. This should either the string "all"
            or a list of maps from ["t2", "t2_err", "m0", "m0_err",
            "r2", "mask"].
        """
        os.makedirs(output_directory, exist_ok=True)
        base_path = os.path.join(output_directory, base_file_name)
        if maps == 'all' or maps == ['all']:
            maps = ['t2', 't2_err', 'm0', 'm0_err', 'r2', 'mask']
        if isinstance(maps, list):
            for result in maps:
                if result == 't2' or result == 't2_map':
                    t2_nifti = nib.Nifti1Image(self.t2_map, affine=self.affine)
                    nib.save(t2_nifti, base_path + '_t2_map.nii.gz')
                elif result == 't2_err':
                    t2_err_nifti = nib.Nifti1Image(self.t2_err,
                                                   affine=self.affine)
                    nib.save(t2_err_nifti, base_path + '_t2_err.nii.gz')
                elif result == 'm0' or result == 'm0_map':
                    m0_nifti = nib.Nifti1Image(self.m0_map, affine=self.affine)
                    nib.save(m0_nifti, base_path + '_m0_map.nii.gz')
                elif result == 'm0_err':
                    m0_err_nifti = nib.Nifti1Image(self.m0_err,
                                                   affine=self.affine)
                    nib.save(m0_err_nifti, base_path + '_m0_err.nii.gz')
                elif result == 'r2' or result == 'r2_map':
                    r2_nifti = nib.Nifti1Image(T2.r2_map(self),
                                               affine=self.affine)
                    nib.save(r2_nifti, base_path + '_r2_map.nii.gz')
                elif result == 'mask':
                    mask_nifti = nib.Nifti1Image(self.mask.astype(int),
                                                 affine=self.affine)
                    nib.save(mask_nifti, base_path + '_mask.nii.gz')
        else:
            raise ValueError('No NIFTI file saved. The variable "maps" '
                             'should be "all" or a list of maps from '
                             '"["t2", "t2_err", "m0", "m0_err", "r2", '
                             '"mask"]".')

        return


def two_param_eq(t, t2, m0):
    """
        Calculate the expected signal from the equation
        signal = M0 * exp(-t / T2)

        Parameters
        ----------
        t: list
            The times the signal will be calculated at
        t2: float
            The T2 of the signal
        m0: float
            The M0 of the signal

        Returns
        -------
        signal: np.ndarray
            The expected signal
        """
    return np.sqrt(np.square(m0 * np.exp(-t / t2)))


def three_param_eq(t, t2, m0, b):
    """
        Calculate the expected signal from the equation
        signal = M0 * exp(-t / T2) + b

        Parameters
        ----------
        t: list
            The times the signal will be calculated at
        t2: float
            The T2 of the signal
        m0: float
            The M0 of the signal
        b: float
            The baseline noise floor of the signal

        Returns
        -------
        signal: np.ndarray
            The expected signal
        """
    return np.sqrt(np.square(m0 * np.exp(-t / t2) + b))
