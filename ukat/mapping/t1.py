import concurrent.futures
import nibabel as nib
import numpy as np
import os
import warnings
from tqdm import tqdm
from scipy.optimize import curve_fit


class T1:
    """
    Attributes
    ----------
    t1_map : np.ndarray
        The estimated T1 values in ms
    t1_err : np.ndarray
        The certainty in the fit of `t1` in ms
    m0_map : np.ndarray
        The estimated M0 values
    m0_err : np.ndarray
        The certainty in the fit of `m0`
    eff_map : np.ndarray
        The estimated inversion efficiency where 0 represents no inversion
        pulse and 2 represents a 180 degree inversion
    eff_err : np.ndarray
        The certainty in the fit of `eff`
    shape : tuple
        The shape of the T1 map
    n_ti : int
        The number of TI used to calculate the map
    """

    def __init__(self, pixel_array, inversion_list, affine, tss=0, tss_axis=-2,
                 mask=None, parameters=2, molli=False, multithread=True):
        """Initialise a T1 class instance.

        Parameters
        ----------
        pixel_array : np.ndarray
            A array containing the signal from each voxel at each inversion
            time with the last dimension being time i.e. the array needed to
            generate a 3D T1 map would have dimensions [x, y, z, TI].
        inversion_list : list()
            An array of the inversion times used for the last dimension of the
            raw data. In milliseconds.
        tss : float, optional
            Default 0
            The temporal slice spacing is the delay between acquisition of
            slices in a T1 map. Including this information means the
            inversion time is correct for each slice in a multi-slice T1
            map. In milliseconds.
        tss_axis : int, optional
            Default -2 i.e. last spatial axis
            The axis over which the temporal slice spacing is applied. This
            axis is relative to the full 4D pixel array i.e. tss_axis=-1
            would be along the TI axis and would be meaningless.
            If `pixel_array` is single slice (dimensions [x, y, TI]),
            then this should be set to None.
        affine : np.ndarray
            A matrix giving the relationship between voxel coordinates and
            world coordinates.
        mask : np.ndarray, optional
            A boolean mask of the voxels to fit. Should be the shape of the
            desired T1 map rather than the raw data i.e. omit the time
            dimension.
        parameters : {2, 3}, optional
            Default `2`
            The number of parameters to fit the data to. A two parameter fit
            will estimate S0 and T1 while a three parameter fit will also
            estimate the inversion efficiency.
        molli : bool, optional
            Default False.
            Apply MOLLI corrections to T1.
        multithread : bool, optional
            Default True.
            If True, fitting will be distributed over all cores available on
            the node. If False, fitting will be carried out on a single thread.
            Multithreading is useful when calculating the T1 for a large
            number of voxels e.g. generating a multi-slice abdominal T1 map.
            Turning off multithreading can be useful when fitting very small
            amounts of data e.g. a mean T1 signal decay over a ROI when the
            overheads of multithreading are more of a hindrance than the
            increase in speed distributing the calculation would generate.
        """

        self.pixel_array = pixel_array
        self.shape = pixel_array.shape[:-1]
        self.dimensions = len(pixel_array.shape)
        self.n_ti = pixel_array.shape[-1]
        self.affine = affine
        # Generate a mask if there isn't one specified
        if mask is None:
            self.mask = np.ones(self.shape, dtype=bool)
        else:
            self.mask = mask
        # Don't process any nan values
        self.mask[np.isnan(np.sum(pixel_array, axis=-1))] = False
        self.inversion_list = inversion_list
        self.tss = tss
        if tss_axis is not None:
            self.tss_axis = tss_axis % self.dimensions
        else:
            self.tss_axis = None
            self.tss = 0
        self.parameters = parameters
        self.molli = molli
        self.multithread = multithread

        # Some sanity checks
        assert (pixel_array.shape[-1]
                == len(inversion_list)), 'Number of inversions does not ' \
                                         'match the number of time frames ' \
                                         'on the last axis of pixel_array'
        if self.tss != 0:
            assert (self.tss_axis != self.dimensions - 1), \
                'Temporal slice spacing can\'t be applied to the TI axis.'
            assert (tss_axis < self.dimensions), \
                'tss_axis must be less than the number of spatial dimensions'
        if self.molli:
            if self.parameters == 2:
                self.parameters = 3
                warnings.warn('MOLLI requires a three parameter fit, '
                              'using parameters=3.')


        # Initialise output attributes
        self.t1_map = np.zeros(self.shape)
        self.t1_err = np.zeros(self.shape)
        self.m0_map = np.zeros(self.shape)
        self.m0_err = np.zeros(self.shape)
        self.eff_map = np.zeros(self.shape)
        self.eff_err = np.zeros(self.shape)

        # Fit data
        if self.parameters == 2:
            self.t1_map, self.t1_err, self.m0_map, self.m0_err = self.__fit__()
        elif self.parameters == 3:
            self.t1_map, self.t1_err, self.m0_map, self.m0_err, \
                self.eff_map, self.eff_err = self.__fit__()
        else:
            raise ValueError('Parameters can be 2 or 3 only. You specified '
                             '{}'.format(self.parameters))

        if self.molli:
            correction_factor = (self.m0_map * self.eff_map) / self.m0_map - 1
            percentage_error = self.t1_err / self.t1_map
            self.t1_map = np.nan_to_num(self.t1_map * correction_factor)
            self.t1_err = np.nan_to_num(self.t1_map * percentage_error)

    def __fit__(self):
        n_vox = np.prod(self.shape)
        # Initialise maps
        t1_map = np.zeros(n_vox)
        m0_map = np.zeros(n_vox)
        t1_err = np.zeros(n_vox)
        m0_err = np.zeros(n_vox)
        if self.parameters == 3:
            eff_map = np.zeros(n_vox)
            eff_err = np.zeros(n_vox)
        mask = self.mask.flatten()
        signal = self.pixel_array.reshape(-1, self.n_ti)
        slices = np.indices(self.shape)[self.tss_axis].ravel()
        # Get indices of voxels to process
        idx = np.argwhere(mask).squeeze()

        # Multithreaded method
        if self.multithread:
            with concurrent.futures.ProcessPoolExecutor() as pool:
                with tqdm(total=idx.size) as progress:
                    futures = []

                    for ind in idx:
                        ti_slice_corrected = self.inversion_list + \
                                             slices[ind] * self.tss
                        future = pool.submit(self.__fit_signal__,
                                             signal[ind, :],
                                             ti_slice_corrected,
                                             self.parameters)
                        future.add_done_callback(lambda p: progress.update())
                        futures.append(future)

                    results = []
                    for future in futures:
                        result = future.result()
                        results.append(result)

            if self.parameters == 2:
                t1_map[idx], t1_err[idx], \
                    m0_map[idx], m0_err[idx] = [np.array(row)
                                                for row in zip(*results)]
            elif self.parameters == 3:
                t1_map[idx], t1_err[idx], \
                    m0_map[idx], m0_err[idx], \
                        eff_map[idx], eff_err[idx] = [np.array(row)
                                                      for row in zip(*results)]

        # Single threaded method
        else:
            with tqdm(total=idx.size) as progress:
                for ind in idx:
                    sig = signal[ind, :]
                    ti_slice_corrected = self.inversion_list + \
                                            slices[ind] * self.tss
                    if self.parameters == 2:
                        t1_map[ind], t1_err[ind], \
                            m0_map[ind], m0_err[ind] = \
                                self.__fit_signal__(sig,
                                                    ti_slice_corrected,
                                                    self.parameters)
                    elif self.parameters == 3:
                        t1_map[ind], t1_err[ind], \
                            m0_map[ind], m0_err[ind], \
                                eff_map[ind], eff_err[ind] = \
                                    self.__fit_signal__(sig,
                                                        ti_slice_corrected,
                                                        self.parameters)
                    progress.update(1)

        # Reshape results to raw data shape
        t1_map = t1_map.reshape(self.shape)
        m0_map = m0_map.reshape(self.shape)
        t1_err = t1_err.reshape(self.shape)
        m0_err = m0_err.reshape(self.shape)

        if self.parameters == 2:
            return t1_map, t1_err, m0_map, m0_err

        elif self.parameters == 3:
            eff_map = eff_map.reshape(self.shape)
            eff_err = eff_err.reshape(self.shape)
            return t1_map, t1_err, m0_map, m0_err, eff_map, eff_err

    def __fit_signal__(self, sig, t, parameters):

        # Initialise parameters and specify equation to fit to
        if parameters == 2:
            bounds = ([0, 0], [5000, 10000000])
            initial_guess = [1000, 30000]
            if sig.min() >= 0:
                eq = two_param_abs_eq
            else:
                eq = two_param_eq
        elif parameters == 3:
            bounds = ([0, 0, 1], [5000, 10000000, 2])
            initial_guess = [1000, 30000, 2]
            if sig.min() >= 0:
                eq = three_param_abs_eq
            else:
                eq = three_param_eq

        # Fit data to equation
        try:
            popt, pcov = curve_fit(eq, t, sig,
                                   p0=initial_guess, bounds=bounds)
        except RuntimeError:
            popt = np.zeros(self.parameters)
            pcov = np.zeros((self.parameters, self.parameters))

        # Extract fits and errors from result variable
        if popt[0] < bounds[1][0] - 1:
            t1 = popt[0]
            m0 = popt[1]
            err = np.sqrt(np.diag(pcov))
            t1_err = err[0]
            m0_err = err[1]
            if self.parameters == 3:
                eff = popt[2]
                eff_err = err[2]
        else:
            t1, m0, t1_err, m0_err = 0, 0, 0, 0
            if self.parameters == 3:
                eff, eff_err = 0, 0

        if self.parameters == 2:
            return t1, t1_err, m0, m0_err
        elif self.parameters == 3:
            return t1, t1_err, m0, m0_err, eff, eff_err

    def r1_map(self):
        """
        Generates the R1 map from the T1 map output by initialising this
        class.

        Parameters
        ----------
        See class attributes in __init__

        Returns
        -------
        r1_map : np.ndarray
            An array containing the R1 map generated
            by the function with R1 measured in ms.
        """
        return np.nan_to_num(np.reciprocal(self.t1_map), posinf=0, neginf=0)

    def to_nifti(self, output_directory=os.getcwd(), base_file_name='Output',
                 maps='all'):
        """Exports some of the T1 class attributes to NIFTI.

        Parameters
        ----------
        output_directory : string, optional
            Path to the folder where the NIFTI files will be saved.
        base_file_name : string, optional
            Filename of the resulting NIFTI. This code appends the extension.
            Eg., base_file_name = 'Output' will result in 'Output.nii.gz'.
        maps : list or 'all', optional
            List of maps to save to NIFTI. This should either the string "all"
            or a list of maps from ["t1", "t1_err", "m0", "m0_err", "eff",
            "eff_err", "r1", "mask"]
        """
        os.makedirs(output_directory, exist_ok=True)
        base_path = os.path.join(output_directory, base_file_name)
        if maps == 'all' or maps == ['all']:
            maps = ['t1', 't1_err', 'm0', 'm0_err', 'eff', 'eff_err', 'r1_map',
                    'mask']
        if isinstance(maps, list):
            for result in maps:
                if result == 't1' or result == 't1_map':
                    t1_nifti = nib.Nifti1Image(self.t1_map, affine=self.affine)
                    nib.save(t1_nifti, base_path + '_t1_map.nii.gz')
                elif result == 't1_err':
                    t1_err_nifti = nib.Nifti1Image(self.t1_err,
                                                   affine=self.affine)
                    nib.save(t1_err_nifti, base_path + '_t1_err.nii.gz')
                elif result == 'm0' or result == 'm0_map':
                    m0_nifti = nib.Nifti1Image(self.m0_map, affine=self.affine)
                    nib.save(m0_nifti, base_path + '_m0_map.nii.gz')
                elif result == 'm0_err':
                    m0_err_nifti = nib.Nifti1Image(self.m0_err,
                                                   affine=self.affine)
                    nib.save(m0_err_nifti, base_path + '_m0_err.nii.gz')
                elif (self.parameters == 3) and \
                     (result == 'eff' or result == 'eff_map'):
                    eff_nifti = nib.Nifti1Image(self.eff_map,
                                                affine=self.affine)
                    nib.save(eff_nifti, base_path + '_eff_map.nii.gz')
                elif self.parameters == 3 and result == 'eff_err':
                    eff_err_nifti = nib.Nifti1Image(self.eff_err,
                                                    affine=self.affine)
                    nib.save(eff_err_nifti, base_path + '_eff_err.nii.gz')
                elif result == 'r1' or result == 'r1_map':
                    r1_nifti = nib.Nifti1Image(T1.r1_map(self),
                                               affine=self.affine)
                    nib.save(r1_nifti, base_path + '_r1_map.nii.gz')
                elif result == 'mask':
                    mask_nifti = nib.Nifti1Image(self.mask.astype(int),
                                                 affine=self.affine)
                    nib.save(mask_nifti, base_path + '_mask.nii.gz')
        else:
            raise ValueError('No NIFTI file saved. The variable "maps" '
                             'should be "all" or a list of maps from '
                             '"["t1", "t1_err", "m0", "m0_err", "eff", '
                             '"eff_err", "r1", "mask"]".')

        return


def two_param_abs_eq(t, t1, m0):
    """
    Calculate the expected signal from the equation signal = abs(M0 * (1 -
    2 * exp(-t / T1)))

    Parameters
    ----------
    t: list
        The times the signal will be calculated at
    t1: float
        The T1 of the signal
    m0: float
        The M0 of the signal

    Returns
    -------
    signal: ndarray
    """
    return np.abs(m0 * (1 - 2 * np.exp(-t / t1)))


def two_param_eq(t, t1, m0):
    """
    Calculate the expected signal from the equation signal = M0 * (1 - 2 *
    exp(-t / T1))

    Parameters
    ----------
    t: list
        The times the signal will be calculated at
    t1: float
        The T1 of the signal
    m0: float
        The M0 of the signal

    Returns
    -------
    signal: ndarray
    """
    return m0 * (1 - 2 * np.exp(-t / t1))


def three_param_abs_eq(t, t1, m0, eff):
    """
    Calculate the expected signal from the equation signal = abs(M0 * (1 -
    eff * exp(-t / T1)))

    Parameters
    ----------
    t: list
        The times the signal will be calculated at
    t1: float
        The T1 of the signal
    m0: float
        The M0 of the signal
    eff: float
        The inversion efficiency (where 0 is no inversion and 2 is a 180
        degree inversion)

    Returns
    -------
    signal: ndarray
    """
    return np.abs(m0 * (1 - eff * np.exp(-t / t1)))


def three_param_eq(t, t1, m0, eff):
    """
    Calculate the expected signal from the equation signal = M0 * (1 - eff *
    exp(-t / T1)))

    Parameters
    ----------
    t: list
        The times the signal will be calculated at
    t1: float
        The T1 of the signal
    m0: float
        The M0 of the signal
    eff: float
        The inversion efficiency (where 0 is no inversion and 2 is a 180
        degree inversion)

    Returns
    -------
    signal: ndarray
    """
    return m0 * (1 - eff * np.exp(-t / t1))


def magnitude_correct(pixel_array):
    """Sign corrects the magnitude of inversion recovery data using the
    complex component of the signal.

    This function uses the methods of Jerzy Szumowski et al
    (https://doi.org/10.1002/jmri.23705).

    Parameters
    ----------
    pixel_array: ndarray
        Can either be a complex array or have the real and imaginary
        parts of the image as the final dimension e.g. a complex 3D image
        could have the dimensions [x, y, z, ti] where [0, 0, 0, 0] = 1 + 2j
        or the dimensions [x, y, z, ti, type] where [0, 0, 0, 0, 0] = 1 and
        [0, 0, 0, 0, 1] = 2.

    Returns
    -------
    corrected_array : ndarray
        An array of the magnitude intensities with signs corrected.
    """

    # Convert data to a complex array if it isn't already
    if not np.iscomplexobj(pixel_array):
        if pixel_array.shape[-1] == 2:
            pixel_array = pixel_array[..., 0] + pixel_array[..., 1] * (0 + 1j)
        else:
            raise ValueError('Last axis of pixel_array must have length 2')

    pixel_array_prime = np.zeros(pixel_array.shape, dtype=np.complex128)

    for ti in range(pixel_array.shape[-1]):
        pixel_array_prime[..., ti] = (pixel_array[..., ti] *
                                      pixel_array[..., -1].conjugate()) \
                                     / np.abs(pixel_array[..., -1])

    phase_factor = np.imag(np.log(pixel_array_prime / np.abs(pixel_array)))
    phase_offset = np.abs(phase_factor) - (np.pi / 2)
    sign = -(phase_offset / np.abs(phase_offset))
    corrected_array = sign * np.abs(pixel_array)
    return corrected_array
