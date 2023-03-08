import inspect
import numpy as np

from pathos.pools import ProcessPool
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from tqdm import tqdm


class Model:
    def __init__(self, pixel_array, x, eq, mask=None, multithread=True):
        """
        A template class for fitting models to pixel arrays.

        Parameters
        ----------
        pixel_array : np.ndarray
            An array containing the signal from each voxel with the last
            dimension being the dependent variable axis
        x : np.ndarray
            An array containing the dependent variable e.g. time
        eq : function
            A function that takes the dependent variable as the first
            argument and the parameters to fit as the remaining arguments
        mask : np.ndarray, optional
            A boolean mask of voxels to fit. Should be the shape of the desired
            map rather than the raw data i.e. omit the dependent variable axis
        multithread : bool, optional
            Default True
            If True, the fitting will be performed in parallel using all
            available cores
        """
        # Attributes that can be set from default inputs
        self.pixel_array = pixel_array
        self.map_shape = pixel_array.shape[:-1]
        self.x = x
        self.eq = eq
        self.mask = mask
        self.multithread = multithread
        self.n_x = pixel_array.shape[-1]
        self.n_params = self._get_n_params()

        # Placeholder attributes that will be overwritten by the child class
        self.initial_guess = None
        self.signal_list = None
        self.x_list = None
        self.p0_list = None
        self.mask_list = None

    def generate_lists(self):
        """
        Generate the lists of data, dependent variables, initial guesses and
        masks to be used in the fitting process
        """
        self.signal_list = self.pixel_array.reshape(-1, self.n_x).tolist()
        self.x_list = [self.x] * len(self.signal_list)
        self.p0_list = [self.initial_guess] * len(self.signal_list)
        self.mask_list = self._get_mask_list()

    def _get_n_params(self):
        """
        Get the number of parameters to fit

        Returns
        -------
        n_params : int
            The number of parameters to fit
        """
        n_params = len(inspect.signature(self.eq).parameters) - 1
        return n_params

    def _get_mask_list(self):
        """
        Get a list of masks to be used in the fitting process, if no mask
        has been specified it will be a list of True i.e. all voxels will be
        fit

        Returns
        -------
        mask_list : list
            A list of booleans indicating whether to fit a voxel or not
        """
        if self.mask is None:
            mask_list = [True] * len(self.signal_list)
            return mask_list
        else:
            mask_list = self.mask.reshape(-1).tolist()
            return mask_list


def fit_image(model):
    """
    Fit an image to a relaxometry curve fitting model

    Parameters
    ----------
    model : ukat.mapping.fitting.relaxation.Model
        A model object containing the data and model to fit to

    Returns
    -------
    popt_list : list
        A list of nD arrays containing the fitted parameters
    error_list : list
        A list of nD arrays containing the error in the fitted parameters
    r2 : np.ndarray
        An nD array containing the R2 value of the fit
    """
    if model.multithread:
        with ProcessPool() as executor:
            results = executor.map(fit_signal,
                                   model.signal_list,
                                   model.x_list,
                                   model.p0_list,
                                   model.mask_list,
                                   [model] * len(model.signal_list))
    else:
        results = list(tqdm(map(fit_signal,
                                model.signal_list,
                                model.x_list,
                                model.p0_list,
                                model.mask_list,
                                [model] * len(model.signal_list)),
                            total=len(model.signal_list)))

    popt_array = np.array([result[0] for result in results])
    popt_list = [popt_array[:, p].reshape(model.map_shape) for p in range(
        model.n_params)]
    error_array = np.array([result[1] for result in results])
    error_list = [error_array[:, p].reshape(model.map_shape) for p in range(
        model.n_params)]
    r2 = np.array([result[2] for result in results]).reshape(model.map_shape)
    return popt_list, error_list, r2


def fit_signal(sig, x, p0, mask, model):
    """
    Fit a signal to a model

    Parameters
    ----------
    sig : np.array
        Numpy array containing the signal to fit
    x : np.array
        Numpy array containing the x values for the signal (e.g. TE)
    p0 : np.array
        Numpy array containing the initial guess for the parameters
    mask : bool
        A boolean indicating whether to fit the signal or not
    model : Model
        A Model object containing the model to fit to

    Returns
    -------
    popt : np.array
        Numpy array containing the fitted parameters
    error : np.array
        Numpy array containing the standard error of the fitted parameters
    r2 : float
        The R^2 value of the fit
    """
    if mask is True:
        try:
            popt, pcov = curve_fit(model.eq, x, sig, p0=p0,
                                   bounds=model.bounds)
            # Remove fits that are hitting (or very close to) the upper bounds
            # as these tend to be inaccurate
            if (popt > np.array(model.bounds[1]) * 0.999).any():
                popt = np.zeros(model.n_params)
                pcov = np.zeros((model.n_params, model.n_params))
        except (RuntimeError, ValueError):
            popt = np.zeros(model.n_params)
            pcov = np.zeros((model.n_params, model.n_params))
    else:
        popt = np.zeros(model.n_params)
        pcov = np.zeros((model.n_params, model.n_params))

    error = np.sqrt(np.diag(pcov))
    fit_sig = model.eq(x, *popt)
    r2 = r2_score(sig, fit_sig)
    return popt, error, r2
