import nibabel as nib
import numpy as np
import os
import warnings

from . import fitting


class T1Model(fitting.Model):
    def __init__(self, pixel_array, ti, parameters=2, mask=None, tss=0,
                 tss_axis=-2, mag_corr=False, multithread=True):
        """
        A class containing the T1 fitting model

        Parameters
        ----------
        pixel_array : np.ndarray
            An array containing the signal from each voxel at each echo
            time with the last dimension being time i.e. the array needed to
            generate a 3D T1 map would have dimensions [x, y, z, TE].
        ti : np.ndarray
            An array of the inversion times used for the last dimension of the
            pixel_array. In milliseconds.
        parameters : {2, 3}, optional
            Default `2`
            The number of parameters to fit the data to. A two parameter fit
            will estimate S0 and T1 while a three parameter fit will also
            estimate the inversion efficiency.
        mask : np.ndarray, optional
            A boolean mask of the voxels to fit. Should be the shape of the
            desired T1 map rather than the raw data i.e. omit the time
            dimension.
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
        mag_corr : {True, False, 'auto'}, optional
            Default False
            If True, the data is assumed to have been magnitude corrected
            using the complex component of the signal and thus negative
            values represent inverted signal. If False, the data will be
            fit to the modulus of the expected signal, negative values are
            simply considered part of the noise in the data. If 'auto',
            the data will be assumed to have been magnitude corrected if 5%
            of the initial inversion time data is negative.
        multithread : bool, optional
            Default True
            If True, the fitting will be performed in parallel using all
            available cores
        """
        self.parameters = parameters
        self.tss = tss
        self.tss_axis = tss_axis

        if mag_corr == 'auto':
            # Find a very rough estimate of the fully recovered signal
            recovered_signal = np.percentile(pixel_array[..., -1], 95)

            # If the fifth percentile of the first inversion time is
            # less than the negative of 5% of the recovered signal
            # then assume the data has been magnitude corrected
            if np.percentile(pixel_array[..., 0], 5) < -recovered_signal * 0.05:
                self.mag_corr = True
                neg_percent = (np.sum(pixel_array[..., 0] < 0)
                               / pixel_array[..., 0].size)
                if neg_percent < 0.05:
                    warnings.warn('Fitting data to a magnitude corrected '
                                  'inversion recovery curve however, less '
                                  'than 5% of the data from the first '
                                  'inversion is negative. If you have '
                                  'performed magnitude correction ignore '
                                  'this warning, otherwise the negative '
                                  'values could be due to noise or '
                                  'preprocessing steps  such as EPI '
                                  'distortion correction and registration.\n'
                                  f'Percentage of first inversion data that '
                                  f'is negative = {neg_percent:.2%}')
            else:
                self.mag_corr = False
                if np.nanmin(pixel_array) < 0:
                    warnings.warn('Negative values found in data from the '
                                  'first inversion but as the first '
                                  'percentile is not negative, it is assumed '
                                  'these are negative due to noise or '
                                  'preprocessing steps such as EPI '
                                  'distortion correction and registration. '
                                  'As such the data will be fit to the '
                                  'modulus of the recovery curve.\n'
                                  f'Min value = '
                                  f'{np.nanmin(pixel_array[..., 0])}\n'
                                  '5th percentile = '
                                  f'{np.percentile(pixel_array[..., 0], 5)}')
        else:
            self.mag_corr = mag_corr

        if self.parameters == 2:
            if self.mag_corr:
                self.t1_eq = two_param_eq
                super().__init__(pixel_array, ti, self.t1_eq, mask,
                                 multithread)
            else:
                self.t1_eq = two_param_abs_eq
                super().__init__(pixel_array, ti, self.t1_eq, mask,
                                 multithread)
            self.bounds = ([0, 0], [5000, 1000000000])
            self.initial_guess = [1000, 30000]
        elif self.parameters == 3:
            if self.mag_corr:
                self.t1_eq = three_param_eq
                super().__init__(pixel_array, ti, self.t1_eq, mask,
                                 multithread)
            else:
                self.t1_eq = three_param_abs_eq
                super().__init__(pixel_array, ti, self.t1_eq, mask,
                                 multithread)
            self.bounds = ([0, 0, 1], [5000, 1000000000, 2])
            self.initial_guess = [1000, 30000, 2]
        else:
            raise ValueError(f'Parameters can be 2 or 3 only. You specified '
                             f'{parameters}.')

        self.generate_lists()
        if self.tss != 0:
            self._tss_correct_ti()

    def _tss_correct_ti(self):
        slices = np.indices(self.map_shape)[self.tss_axis].ravel()
        for ind, (ti, slice) in enumerate(zip(self.x_list, slices)):
            self.x_list[ind] = np.array(ti) + self.tss * slice


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
    r2 : np.ndarray
        The R-Squared value of the fit, values close to 1 indicate a good
        fit, lower values indicate a poorer fit
    shape : tuple
        The shape of the T1 map
    n_ti : int
        The number of TI used to calculate the map
    n_vox : int
        The number of voxels in the map i.e. the product of all dimensions
        apart from TI
    """

    def __init__(self, pixel_array, inversion_list, affine, tss=0, tss_axis=-2,
                 mask=None, parameters=2, mag_corr=False, molli=False,
                 multithread=True):
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
        mag_corr : {True, False, 'auto'}, optional
            Default False
            If True, the data is assumed to have been magnitude corrected
            using the complex component of the signal and thus negative
            values represent inverted signal. If False, the data will be
            fit to the modulus of the expected signal, negative values are
            simply considered part of the noise in the data. If 'auto',
            the data will be assumed to have been magnitude corrected if 5%
            of the initial inversion time data is negative.
        molli : bool, optional
            Default False.
            Apply MOLLI corrections to T1.
        multithread : bool or 'auto', optional
            Default 'auto'.
            If True, fitting will be distributed over all cores available on
            the node. If False, fitting will be carried out on a single thread.
            Multithreading is useful when calculating the T1 for a large
            number of voxels e.g. generating a multi-slice abdominal T1 map.
            Turning off multithreading can be useful when fitting very small
            amounts of data e.g. a mean T1 signal decay over a ROI when the
            overheads of multithreading are more of a hindrance than the
            increase in speed distributing the calculation would generate.
            'auto' attempts to apply multithreading where appropriate based
            on the number of voxels being fit.
        """
        assert multithread in [True, False,
                               'auto'], \
            (f'multithreaded must be True, False or auto. '
             f'You entered {multithread}')
        assert mag_corr in [True, False,
                               'auto'], \
            (f'mag_corr must be True, False or auto. '
             f'You entered {mag_corr}')

        self.pixel_array = pixel_array
        self.shape = pixel_array.shape[:-1]
        self.dimensions = len(pixel_array.shape)
        self.n_ti = pixel_array.shape[-1]
        self.n_vox = np.prod(self.shape)
        self.affine = affine
        # Generate a mask if there isn't one specified
        if mask is None:
            self.mask = np.ones(self.shape, dtype=bool)
        else:
            self.mask = mask.astype(bool)
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
        self.mag_corr = mag_corr
        self.molli = molli
        if multithread == 'auto':
            if self.n_vox > 20:
                multithread = True
            else:
                multithread = False
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

        # Fit Data
        self.fitting_model = T1Model(self.pixel_array, self.inversion_list,
                                     self.parameters, self.mask, self.tss,
                                     self.tss_axis, self.mag_corr,
                                     self.multithread)
        self.mag_corr = self.fitting_model.mag_corr
        popt, error, r2 = fitting.fit_image(self.fitting_model)
        self.t1_map = popt[0]
        self.m0_map = popt[1]
        self.t1_err = error[0]
        self.m0_err = error[1]
        self.r2 = r2

        if self.parameters == 3:
            self.eff_map = popt[2]
            self.eff_err = error[2]

        # Filter values that are very close to models upper bounds of T1 or
        # M0 out. Not filtering based on eff as this should ideally be at
        # the upper bound!
        threshold = 0.999  # 99.9% of the upper bound
        bounds_mask = ((self.t1_map > self.fitting_model.bounds[1][0] *
                        threshold) |
                       (self.m0_map > self.fitting_model.bounds[1][1] *
                        threshold))
        self.t1_map[bounds_mask] = 0
        self.m0_map[bounds_mask] = 0
        self.t1_err[bounds_mask] = 0
        self.m0_err[bounds_mask] = 0
        self.r2[bounds_mask] = 0
        if self.parameters == 3:
            self.eff_map[bounds_mask] = 0
            self.eff_err[bounds_mask] = 0

        # Do MOLLI correction
        if self.molli:
            correction_factor = (self.m0_map * self.eff_map) / self.m0_map - 1
            percentage_error = self.t1_err / self.t1_map
            self.t1_map = np.nan_to_num(self.t1_map * correction_factor)
            self.t1_err = np.nan_to_num(self.t1_map * percentage_error)

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
        with np.errstate(divide='ignore'):
            r1_map = np.nan_to_num(np.reciprocal(self.t1_map), posinf=0,
                                   neginf=0)
        return r1_map

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
            "eff_err", "r1", "r2", "mask"]
        """
        os.makedirs(output_directory, exist_ok=True)
        base_path = os.path.join(output_directory, base_file_name)
        if maps == 'all' or maps == ['all']:
            maps = ['t1', 't1_err', 'm0', 'm0_err', 'eff', 'eff_err', 'r1_map',
                    'r2', 'mask']
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
                elif result == 'r2':
                    r2_nifti = nib.Nifti1Image(self.r2,
                                               affine=self.affine)
                    nib.save(r2_nifti, base_path + '_r2.nii.gz')
                elif result == 'mask':
                    mask_nifti = nib.Nifti1Image(self.mask.astype(np.uint16),
                                                 affine=self.affine)
                    nib.save(mask_nifti, base_path + '_mask.nii.gz')
        else:
            raise ValueError('No NIFTI file saved. The variable "maps" '
                             'should be "all" or a list of maps from '
                             '"["t1", "t1_err", "m0", "m0_err", "eff", '
                             '"eff_err", "r1", "mask"]".')

        return

    def get_fit_signal(self):
        """
        Get the fit signal from the model used to fit the data i.e. the
        simulated signal at each inversion time given the estimated T1, M0
        (and inversion efficiency if applicable).

        Returns
        -------
        fit_signal : np.ndarray
            An array containing the fit signal generated by the model
        """
        if self.molli:
            t1 = self.t1_map / ((self.m0_map * self.eff_map) / self.m0_map - 1)
        else:
            t1 = self.t1_map
        t1_lin = t1.reshape(-1)
        m0_lin = self.m0_map.reshape(-1)
        if self.parameters == 3:
            eff_lin = self.eff_map.reshape(-1)

        fit_signal = np.zeros((self.n_vox, self.n_ti))

        if self.parameters == 2:
            for n, (ti, t1, m0) in enumerate(zip(self.fitting_model.x_list,
                                                 t1_lin,
                                                 m0_lin)):
                fit_signal[n, :] = self.fitting_model.t1_eq(ti, t1, m0)
        else:
            for n, (ti, t1, m0, eff) in (
                enumerate(zip(self.fitting_model.x_list,
                              t1_lin,
                              m0_lin,
                              eff_lin))):
                fit_signal[n, :] = self.fitting_model.t1_eq(ti, t1, m0, eff)

        fit_signal = fit_signal.reshape((*self.shape, self.n_ti))
        return fit_signal


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
    with np.errstate(divide='ignore'):
        signal = np.abs(m0 * (1 - 2 * np.exp(-t / t1)))
    return signal


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
    with np.errstate(divide='ignore'):
        signal = m0 * (1 - 2 * np.exp(-t / t1))
    return signal


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
    with np.errstate(divide='ignore'):
        signal = np.abs(m0 * (1 - eff * np.exp(-t / t1)))
    return signal


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
    with np.errstate(divide='ignore'):
        signal = m0 * (1 - eff * np.exp(-t / t1))
    return signal


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
