import numpy as np
from multiprocessing import cpu_count
import concurrent.futures
from tqdm import tqdm
from scipy.optimize import curve_fit


class T2Star(object):
    """Package containing algorithms that calculate parameter maps
    of the MRI scans acquired during the UKRIN-MAPS project.

    Attributes
    ----------
    t2star_map : np.ndarray
        The estimated T2* values in ms
    m0_map : np.ndarray
        The estimated M0 values
    shape : tuple
        The shape of the T2* map
    n_te : int
        The number of TE used to calculate the map

    """

    def __init__(self, pixel_array, echo_list, mask=None, method='loglin',
                 multithread=True):
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
        mask : np.ndarray, optional
            A boolean mask of the voxels to fit. Should be the shape of the
            desired T1 map rather than the raw data i.e. omit the time
            dimension.
        method : {'loglin', '2p_exp'}, optional
            The method used to estimate T2* values. `loglin` is far quicker
            but produces inaccurate results for T2* below 20 ms. `2p_exp` is
            accurate below 20 ms however this comes at the expense of run time.
            Default `loglin`
        multithread : bool, optional
            Default True.
            If True, fitting will be distributed over all cores available on
            the node. If False, fitting will be carried out on a single thread.
            Multithreading is useful when calculating the T2* for a large
            number of voxels e.g. generating a multi-slice abdominal T2* map.
            Turning off multithreading can be useful when fitting very small
            amounts of data e.g. a mean T2* signal decay over an ROI when the
            overheads of multi-threading are more of a hindrance than the
            increase in speed distributing the calculation would generate.
        """
        # Some sanity checks
        assert (pixel_array.shape[-1]
                == len(echo_list)), 'Number of echoes does not match the ' \
                                    'number of time frames on the last axis ' \
                                    'of pixel_array'
        assert method == ('loglin' or '2p_exp'), 'method must be  ' \
                                                 '\'loglin\' or \'2p_exp\''
        self.pixel_array = pixel_array
        self.shape = pixel_array.shape[:-1]
        self.n_te = pixel_array.shape[-1]
        # Generate a mask if there isn't one specified
        if mask is None:
            self.mask = np.ones(self.shape, dtype=bool)
        else:
            self.mask = mask
            # Don't process any nan values
        self.mask[np.isnan(np.sum(pixel_array, axis=-1))] = False
        self.echo_list = echo_list
        self.method = method
        self.multithread = multithread

        # Fit data
        self.t2star_map, self.m0_map = self.__fit__()

    def __fit__(self):
        n_vox = np.prod(self.shape)
        # Initialise maps
        t2star_map = np.zeros(n_vox)
        m0_map = np.zeros(n_vox)
        mask = self.mask.flatten()
        signal = self.pixel_array.reshape(-1, self.n_te)
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
                                             self.echo_list,
                                             self.method)
                        future.add_done_callback(lambda p: progress.update())
                        futures.append(future)

                    results = []
                    for future in futures:
                        result = future.result()
                        results.append(result)
            t2star_map[idx], m0_map[idx] = [np.array(row) for row in zip(
                *results)]

        # Single threaded method
        else:
            with tqdm(total=idx.size) as progress:
                for ind in idx:
                    sig = signal[ind, :]
                    t2star_map[ind], m0_map[ind] = \
                        self.__fit_signal__(sig, self.echo_list, self.method)
                    progress.update(1)

        # Reshape results to raw data shape
        t2star_map = t2star_map.reshape(self.shape)
        m0_map = m0_map.reshape(self.shape)

        return t2star_map, m0_map

    def __fit_signal__(self, sig, te, method):

        self.pixel_array[self.pixel_array == 0] = 1E-10

        if method == 'loglin':
            with np.errstate(invalid='ignore', over='ignore'):
                noise = 0.0
                sd = 0.0
                s_w = 0.0
                s_wx = 0.0
                s_wx2 = 0.0
                s_wy = 0.0
                s_wxy = 0.0
                n_te = len(sig)

                for t in range(n_te):
                    noise = noise + sig[t]
                    sd = sd + sig[t] ** 2
                noise = noise / n_te
                sd = sd / n_te - noise ** 2
                sd = sd ** 2
                sd = np.sqrt(sd)

                for t in range(n_te):
                    if sig[t] > 0:
                        te_tmp = te[t]
                        if sig[t] > sd:
                            sigma = np.log(sig[t] / (sig[t] - sd))
                        else:
                            sigma = np.loc(sig[t] / 0.0001)
                        logsig = np.log(sig)
                        weight = 1 / sigma ** 2

                        s_w += weight
                        s_wx += weight * te_tmp
                        s_wx2 += weight * te_tmp ** 2
                        s_wy += weight * logsig
                        s_wxy += weight * te_tmp * logsig

                delta = (s_w * s_wx2) - (s_wx ** 2)
                a = (1 / delta) * (s_wx2 * s_wy - s_wx * s_wxy)
                b = (1 / delta) * (s_w * s_wxy - s_wx * s_wy)
                t2star = np.real(-1 / b)
                m0 = np.real(np.exp(a))
        elif method == '2p_exp':
            # Initialise parameters
            bounds = ([0, 0], [700, 10])
            initial_guess = [20, 1]

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

        return t2star, m0

    def r2star(self):
        """
        Generates the R2* map from the T2* map output by initialising this
        class.

        Parameters
        ----------
        See class attributes in __init__

        Returns
        -------
        r2star : ndarray
            An array containing the R2* map generated
            by the function with R2* measured in seconds.
        """
        r2star = np.reciprocal(self.t2star_map)
        return r2star


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
        signal: ndarray
        """
    return np.sqrt(np.square(m0 * np.exp(-t / t2star)))
