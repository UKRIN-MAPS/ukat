import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from scipy.optimize import curve_fit


class T1(object):
    """Package containing algorithms that calculate parameter maps
    of the MRI scans acquired during the UKRIN-MAPS project.

    Attributes
    ----------
    See parameters of __init__ class

    """

    def __init__(self, pixel_array, inversion_list, mask=None, parameters=2,
                 multithread=True, chunksize=500):
        """Initialise a T1 class instance.

        Parameters
        ----------
        pixel_array : np.ndarray
            A array containing the signal from each voxel at each inversion
            time with the last dimension being time i.e. the array needed to
            generate a 3D T1 map would have dimensions [x, y, z, TI].
        inversion_list : list()
            An array of the inversion times used for the last dimension of the
            raw data.
        mask : np.ndarray
            A boolean mask of the voxels to fit. Should be the shape of the
            desired T1 map rather than the raw data i.e. omit the time
            dimension.
        parameters : int (2 or 3)
            The number of parameters to fit the data to. A two parameter fit
            will estimate S0 and T1 while a three parameter fit will also
            estimate the inversion efficiency.
        multithread : bool
            If True, fitting will be distributed over all cores available on
            the node. If False, fitting will be carried out on a single thread.
            Useful when fitting very small amounts of data e.g. a mean T1
            decay over an ROI when the overheads of multi-threading are more
            of a hindrance than the increase in speed distributing the
            calculation would generate.
        """

        # Some sanity checks
        assert (pixel_array.shape[-1]
                == len(inversion_list)), 'Number of inversions does not ' \
                                         'match the number of time frames ' \
                                         'on the last axis of pixel_array'
        self.pixel_array = pixel_array
        self.shape = pixel_array.shape[:-1]
        self.n_ti = pixel_array.shape[-1]
        if mask is None:
            self.mask = np.ones(self.shape, dtype=bool)
        else:
            self.mask = mask
        self.inversion_list = inversion_list
        self.parameters = parameters
        self.multithread = multithread
        self.chunksize = chunksize
        if self.parameters == 2:
            self.t1_map, self.t1_err, self.m0_map, self.m0_err = self.__fit__()
        elif self.parameters == 3:
            self.t1_map, self.t1_err, self.m0_map, self.m0_err, \
             self.eff, self.eff_err = self.__fit__()
        else:
            raise ValueError('Parameters can be 2 or 3 only. You specified '
                             '{}'.format(self.parameters))
        self.r1_map = np.reciprocal(self.t1_map)

    def __fit__(self):
        n_vox = np.prod(self.shape)
        t1_map = np.zeros(n_vox)
        m0_map = np.zeros(n_vox)
        t1_err = np.zeros(n_vox)
        m0_err = np.zeros(n_vox)
        if self.parameters == 3:
            eff_map = np.zeros(n_vox)
            eff_err = np.zeros(n_vox)
        mask = self.mask.flatten()
        signal = self.pixel_array.reshape(-1, self.n_ti)
        idx = np.argwhere(mask).squeeze()
        
        if self.multithread:
            fit_partial = partial(self.__fit_wrapper__, signal,
                                  self.inversion_list,
                                  self.parameters)
            with Pool() as pool:
                res = list(tqdm(pool.imap(fit_partial, idx, self.chunksize),
                total=idx.size, leave=False))
            res_array = np.array(res)
            t1_map[idx] = res_array[:, 0]
        else:
            for ind in idx:
                sig = signal[ind, :]
                output_tuple = self.__fit_signal__(sig, self.inversion_list, self.parameters)
                t1_map[ind] = output_tuple[0]
                m0_map[ind] = output_tuple[1]
                t1_err[ind] = output_tuple[2]
                m0_err[ind] = output_tuple[3]
                if self.parameters == 3:
                    eff_map = output_tuple[4]
                    eff_err = output_tuple[5]
        
        t1_map = t1_map.reshape(self.shape)
        m0_map = m0_map.reshape(self.shape)
        t1_err = t1_err.reshape(self.shape)
        m0_err = m0_err.reshape(self.shape)
        if self.parameters == 2:
            return t1_map, t1_err, m0_map, m0_err
        elif self.parameters == 3:
            eff_map = eff_map.reshape(self.shape)
            eff_err = eff_err.reshape(self.shape)
            return t1_map, t1_err, m0_map, m0_err, eff, eff_err

    @staticmethod
    def __two_param_abs_eq__(t, t1, m0):
        return np.abs(m0 * 1 - 2 * np.exp(-t / t1))

    @staticmethod
    def __two_param_eq__(t, t1, m0):
        return m0 * 1 - 2 * np.exp(-t / t1)

    @staticmethod
    def __three_param_abs_eq__(t, t1, m0, eff):
        return np.abs(m0 * (1 - eff * np.exp(-t / t1)))

    @staticmethod
    def __three_param_eq__(t, t1, m0, eff):
        return m0 * (1 - eff * np.exp(-t / t1))

    def __fit_wrapper__(self, idx, signal, ti, parameters):
        signal_selected  = signal[idx.astype(np.int8)] # This should be something like [idx, :] but that seems to be causing problems so has been removed for now
        output_tuple = self.__fit_signal__(signal_selected, ti, parameters)
        return output_tuple

    def __fit_signal__(self, sig, t, parameters):
        if parameters == 2:
            bounds = ([0, 0], [4000, 1000000])
            initial_guess = [1000, 30000]
            if sig.min() > 0:
                eq = self.__two_param_abs_eq__
            else:
                eq = self.__two_param_eq__
        elif parameters == 3:
            bounds = ([0, 0, 0], [4000, 1000000, 2])
            initial_guess = [1000, 30000, 2]
            if sig.min() > 0:
                eq = self.__three_param_abs_eq__
            else:
                eq = self.__three_param_eq__

        try:
            popt, pcov = curve_fit(eq, t, sig,
                                   p0=initial_guess,  bounds=bounds)
        except RuntimeError:
            popt = np.zeros(self.parameters)
            pcov = np.zeros((self.parameters, self.parameters))

        if popt[0] < bounds[1][0]:
            t1 = popt[0]
            m0 = popt[1]
            err = np.sqrt(np.diag(pcov))
            t1_err = err[0]
            m0_err = err[1]
            output_tuple = tuple([t1, t1_err, m0, m0_err])
            if self.parameters == 3:
                eff = popt[2]
                eff_err = err[2]
                output_tuple = tuple([t1, t1_err, m0, m0_err, eff, eff_err])
        else:
            t1, m0, t1_err, m0_err = 0, 0, 0, 0
            output_tuple = tuple([t1, t1_err, m0, m0_err])
            if self.parameters == 3:
                eff, eff_err = 0, 0
                output_tuple = tuple([t1, t1_err, m0, m0_err, eff, eff_err])
        return output_tuple


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
