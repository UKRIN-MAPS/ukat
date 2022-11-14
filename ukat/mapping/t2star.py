import os
import warnings
import numpy as np
import nibabel as nib
import concurrent.futures
from tqdm import tqdm
from scipy.optimize import curve_fit


class T2Star:
    """
    Attributes
    ----------
    t2star_map : np.ndarray
        The estimated T2* values in ms
    t2star_err : np.ndarray
        The certainty in the fit of `t2star_map` in ms. Only returned if
        `2p_exp` method is used, otherwise is an array of nan
    m0_map : np.ndarray
        The estimated M0 values
    m0_err : np.ndarray
        The certainty in the fit of `m0_map`. Only returned if `2p_exp`
        method is used, otherwise is an array of nan
    shape : tuple
        The shape of the T2* map
    n_te : int
        The number of TE used to calculate the map
    n_vox : int
        The number of voxels in the map i.e. the product of all dimensions
        apart from TE
    """

    def __init__(self, pixel_array, echo_list, affine, mask=None,
                 method='loglin', multithread='auto'):
        """Initialise a T2Star class instance.

        Parameters
        ----------
        pixel_array : np.ndarray
            A array containing the signal from each voxel at each echo
            time with the last dimension being time i.e. the array needed to
            generate a 3D T2* map would have dimensions [x, y, z, TE].
        echo_list : list()
            An array of the echo times used for the last dimension of the
            raw data. In milliseconds.
        affine : np.ndarray
            A matrix giving the relationship between voxel coordinates and
            world coordinates.
        mask : np.ndarray, optional
            A boolean mask of the voxels to fit. Should be the shape of the
            desired T2* map rather than the raw data i.e. omit the time
            dimension.
        method : {'loglin', '2p_exp'}, optional
            Default `loglin`
            The method used to estimate T2* values. 'loglin' uses a
            weighted linear fit to the natural logarithm of the
            signal. '2p_exp' fits the signal to a two parameter
            exponential (S = S0 * exp(-t / T2*)). `loglin` is far quicker
            but produces inaccurate results for T2* below 20 ms. `2p_exp` is
            accurate below 20 ms however this comes at the expense of run time.
        multithread : bool or 'auto', optional
            Default 'auto'.
            If True, fitting will be distributed over all cores available on
            the node. If False, fitting will be carried out on a single thread.
            'auto' attempts to apply multithreading where appropriate based
            on the number of voxels being fit and the method being used.
            Generally 'loglin' is quicker running single threaded  due to
            the additional overheads of multithreading while '2p_exp' is
            quicker running multithreaded for anything but small numbers of
            voxels.
        """
        # Some sanity checks
        assert (pixel_array.shape[-1]
                == len(echo_list)), 'Number of echoes does not match the ' \
                                    'number of time frames on the last axis ' \
                                    'of pixel_array'
        assert method == 'loglin' \
            or method == '2p_exp', 'method must be loglin or 2p_exp. You ' \
                                   'entered {}'.format(method)
        assert multithread is True \
            or multithread is False \
            or multithread == 'auto', 'multithreaded must be True, False or ' \
                                      'auto. You entered {}'\
                                      .format(multithread)
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
        self.echo_list = echo_list
        self.method = method
        # Auto multithreading conditions
        if multithread == 'auto':
            if self.method == '2p_exp' and self.n_vox > 20:
                multithread = True
            else:
                multithread = False
        self.multithread = multithread

        # Fit data
        self.t2star_map, self.t2star_err, self.m0_map, self.m0_err\
            = self.__fit__()

        # Warn if using loglin method to produce a map with a large
        # proportion of T2* < 20 ms i.e. where loglin isn't as accurate.
        if self.method == 'loglin':
            proportion_less_than_20 = np.sum((self.t2star_map > 0) &
                                             (self.t2star_map < 20)) \
                                      / np.prod(self.n_vox)
            warn_thresh = 0.3
            if proportion_less_than_20 > warn_thresh:
                warnings.warn('{:%} of voxels in this map have a T2* less '
                              'than 20 ms. The loglin method is not accurate '
                              'in this regime. If these voxels are of '
                              'interest, consider using the 2p_exp fitting'
                              ' method'.format(proportion_less_than_20))

    def __fit__(self):

        # Initialise maps
        t2star_map = np.zeros(self.n_vox)
        t2star_err = np.zeros(self.n_vox)
        m0_map = np.zeros(self.n_vox)
        m0_err = np.zeros(self.n_vox)
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
                        future = pool.submit(self.__fit_signal__,
                                             signal[ind, :],
                                             self.echo_list,
                                             self.method)
                        future.add_done_callback(lambda p: progress.update())
                        futures.append(future)

                    results = []
                    for future in futures:
                        result = future.result()
                        results.append(result)
            t2star_map[idx], t2star_err[idx], m0_map[idx], m0_err[idx] \
                = [np.array(row) for row in zip(*results)]

        # Single threaded method
        else:
            with tqdm(total=idx.size) as progress:
                for ind in idx:
                    sig = signal[ind, :]
                    t2star_map[ind], t2star_err[ind], \
                        m0_map[ind], m0_err[ind] \
                        = self.__fit_signal__(sig, self.echo_list, self.method)
                    progress.update(1)

        # Reshape results to raw data shape
        t2star_map = t2star_map.reshape(self.shape)
        t2star_err = t2star_err.reshape(self.shape)
        m0_map = m0_map.reshape(self.shape)
        m0_err = m0_err.reshape(self.shape)

        return t2star_map, t2star_err, m0_map, m0_err

    @staticmethod
    def __fit_signal__(sig, te, method):
        if method == 'loglin':
            s_w = 0.0
            s_wx = 0.0
            s_wx2 = 0.0
            s_wy = 0.0
            s_wxy = 0.0
            n_te = len(sig)

            noise = sig.sum() / n_te
            sd = np.abs(np.sum(sig ** 2) / n_te - noise ** 2)
            if sd > 1e-10:
                for t in range(n_te):
                    if sig[t] > 0:
                        te_tmp = te[t]
                        if sig[t] > sd:
                            sigma = np.log(sig[t] / (sig[t] - sd))
                        else:
                            sigma = np.log(sig[t] / 0.0001)
                        logsig = np.log(sig[t])
                        weight = 1 / sigma ** 2

                        s_w += weight
                        s_wx += weight * te_tmp
                        s_wx2 += weight * te_tmp ** 2
                        s_wy += weight * logsig
                        s_wxy += weight * te_tmp * logsig

                delta = (s_w * s_wx2) - (s_wx ** 2)
                if delta > 1e-5:
                    a = (1 / delta) * (s_wx2 * s_wy - s_wx * s_wxy)
                    b = (1 / delta) * (s_w * s_wxy - s_wx * s_wy)
                    t2star = np.real(-1 / b)
                    m0 = np.real(np.exp(a))
                    if t2star < 0 or t2star > 700 or np.isnan(t2star):
                        t2star = 0
                        m0 = 0
                else:
                    t2star = 0
                    m0 = 0
            else:
                t2star = 0
                m0 = 0
            t2star_err = np.nan
            m0_err = np.nan

        elif method == '2p_exp':
            # Initialise parameters
            bounds = ([0, 0], [700, 100000000])
            initial_guess = [20, 10000]

            # Fit data to equation
            try:
                popt, pcov = curve_fit(two_param_eq, te, sig,
                                       p0=initial_guess, bounds=bounds)
            except RuntimeError:
                popt = np.zeros(2)
                pcov = np.zeros((2, 2))

            # Extract fits and errors from result variables
            if popt[0] < bounds[1][0] - 1:
                t2star = popt[0]
                m0 = popt[1]
                err = np.sqrt(np.diag(pcov))
                t2star_err = err[0]
                m0_err = err[1]
            else:
                t2star, m0, t2star_err, m0_err = 0, 0, 0, 0

        return t2star, t2star_err, m0, m0_err

    def r2star_map(self):
        """
        Generates the R2* map from the T2* map output by initialising this
        class.

        Parameters
        ----------
        See class attributes in __init__

        Returns
        -------
        r2star_map : np.ndarray
            An array containing the R2* map generated
            by the function with R2* measured in ms.
        """
        return np.nan_to_num(np.reciprocal(self.t2star_map),
                             posinf=0, neginf=0)

    def to_nifti(self, output_directory=os.getcwd(), base_file_name='Output',
                 maps='all'):
        """Exports some of the T2Star class attributes to NIFTI.
                
        Parameters
        ----------
        output_directory : string, optional
            Path to the folder where the NIFTI files will be saved.
        base_file_name : string, optional
            Filename of the resulting NIFTI. This code appends the extension.
            Eg., base_file_name = 'Output' will result in 'Output.nii.gz'.
        maps : list or 'all', optional
            List of maps to save to NIFTI. This should either the string "all"
            or a list of maps from ["t2star", "t2star_err", "m0",
            "m0_err", "r2star", "mask"].
        """
        os.makedirs(output_directory, exist_ok=True)
        base_path = os.path.join(output_directory, base_file_name)
        if maps == 'all' or maps == ['all']:
            maps = ['t2star', 'm0', 'r2star', 'mask']
            if self.method == '2p_exp':
                maps += ['t2star_err', 'm0_err']
        if isinstance(maps, list):
            for result in maps:
                if result == 't2star' or result == 't2star_map':
                    t2star_nifti = nib.Nifti1Image(self.t2star_map,
                                                   affine=self.affine)
                    nib.save(t2star_nifti, base_path + '_t2star_map.nii.gz')
                elif result == 't2star_err' or result == 't2star_err_map':
                    t2star_err_nifti = nib.Nifti1Image(self.t2star_err,
                                                       affine=self.affine)
                    nib.save(t2star_err_nifti, base_path +
                             '_t2star_err.nii.gz')
                    if self.method == 'loglin':
                        warnings.warn('Saving t2star_error however, '
                                      'the loglin method does not produce '
                                      'confidence intervals. As such the '
                                      'resulting nifti will be all zeros.')
                elif result == 'm0' or result == 'm0_map':
                    m0_nifti = nib.Nifti1Image(self.m0_map, affine=self.affine)
                    nib.save(m0_nifti, base_path + '_m0_map.nii.gz')
                elif result == 'm0_err' or result == 'm0_err_map':
                    m0_err_nifti = nib.Nifti1Image(self.m0_err,
                                                   affine=self.affine)
                    nib.save(m0_err_nifti, base_path + '_m0_err.nii.gz')
                    if self.method == 'loglin':
                        warnings.warn('Saving m0_error however, the loglin '
                                      'method does not produce confidence '
                                      'intervals. As such the resulting nifti'
                                      ' will be all zeros.')
                elif result == 'r2star' or result == 'r2star_map':
                    r2star_nifti = nib.Nifti1Image(T2Star.r2star_map(self),
                                                   affine=self.affine)
                    nib.save(r2star_nifti, base_path + '_r2star_map.nii.gz')
                elif result == 'mask':
                    mask_nifti = nib.Nifti1Image(self.mask.astype(int),
                                                 affine=self.affine)
                    nib.save(mask_nifti, base_path + '_mask.nii.gz')
        else:
            raise ValueError('No NIFTI file saved. The variable "maps" '
                             'should be "all" or a list of maps from '
                             '"["t2star", "t2star_err", "m0", "m0_err", '
                             '"r2star", "mask"]".')

        return


def two_param_eq(t, t2star, m0):
    """
        Calculate the expected signal from the equation
        signal = M0 * exp(-t / T2*)

        Parameters
        ----------
        t: list
            The times the signal will be calculated at
        t2star: float
            The T2* of the signal
        m0: float
            The M0 of the signal

        Returns
        -------
        signal: np.ndarray
        """
    return np.sqrt(np.square(m0 * np.exp(-t / t2star)))
