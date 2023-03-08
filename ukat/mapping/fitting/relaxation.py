import numpy as np

from pathos.pools import ProcessPool
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from tqdm import tqdm


def fit_image(model):
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
