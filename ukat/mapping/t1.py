import numpy as np
from multiprocessing import cpu_count
import concurrent.futures
from tqdm import tqdm
from scipy.optimize import curve_fit


class T1(object):
    """Package containing algorithms that calculate parameter maps
    of the MRI scans acquired during the UKRIN-MAPS project.

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
        The certianty in the fit of `eff`
    r1_map : np.ndarray
        The estimated R1 map in ms^-1
    shape : tuple
        The shape of the T1 map
    n_ti : int
        The number of TI used to calculate the map
    """

    def __init__(self, pixel_array, inversion_list, mask=None, parameters=2,
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
        mask : np.ndarray
            A boolean mask of the voxels to fit. Should be the shape of the
            desired T1 map rather than the raw data i.e. omit the time
            dimension.
        parameters : {2, 3}
            The number of parameters to fit the data to. A two parameter fit
            will estimate S0 and T1 while a three parameter fit will also
            estimate the inversion efficiency.
        multithread : bool, optional
            Default True.
            If True, fitting will be distributed over all cores available on
            the node. If False, fitting will be carried out on a single thread.
            Multithreading is useful when calculating the T1 for a large
            number of voxels e.g. generating a multi-slice abdominal T1 map.
            Turning off multithreading can be useful when fitting very small
            amounts of data e.g. a mean T1 signal decay over an ROI when the
            overheads of multi-threading are more of a hindrance than the
            increase in speed distributing the calculation would generate.
        """

        # Some sanity checks
        assert (pixel_array.shape[-1]
                == len(inversion_list)), 'Number of inversions does not ' \
                                         'match the number of time frames ' \
                                         'on the last axis of pixel_array'
        self.pixel_array = pixel_array
        self.shape = pixel_array.shape[:-1]
        self.n_ti = pixel_array.shape[-1]
        # Generate a mask if there isn't one specified
        if mask is None:
            self.mask = np.ones(self.shape, dtype=bool)
        else:
            self.mask = mask
        # Don't process any nan values
        self.mask[np.isnan(np.sum(pixel_array, axis=-1))] = False
        self.inversion_list = inversion_list
        self.parameters = parameters
        self.multithread = multithread

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
        # Get indices of voxels to process
        idx = np.argwhere(mask).squeeze()

        # Multithreaded method
        if self.multithread:
            cores = cpu_count()
            with concurrent.futures.ProcessPoolExecutor(cores) as pool:
                with tqdm(total=idx.size) as progress:
                    futures = []

                    for ind in idx:
                        future = pool.submit(self.__fit_signal__,
                                             signal[ind, :],
                                             self.inversion_list,
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
                    if self.parameters == 2:
                        t1_map[ind], t1_err[ind], \
                            m0_map[ind], m0_err[ind] = \
                                self.__fit_signal__(sig,
                                                    self.inversion_list,
                                                    self.parameters)
                    elif self.parameters == 3:
                        t1_map[ind], t1_err[ind], \
                            m0_map[ind], m0_err[ind], \
                                eff_map[ind], eff_err[ind] = \
                                    self.__fit_signal__(sig,
                                                        self.inversion_list,
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
            bounds = ([0, 0], [4000, 1000000])
            initial_guess = [1000, 30000]
            if sig.min() > 0:
                eq = two_param_abs_eq
            else:
                eq = two_param_eq
        elif parameters == 3:
            bounds = ([0, 0, 0], [4000, 1000000, 2])
            initial_guess = [1000, 30000, 2]
            if sig.min() > 0:
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
        r1_map = np.reciprocal(self.t1_map)
        return r1_map


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
