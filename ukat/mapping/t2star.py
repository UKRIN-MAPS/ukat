import os
import warnings
import numpy as np
import nibabel as nib

from . import fitting

from pathos.pools import ProcessPool
from tqdm import tqdm
from sklearn.metrics import r2_score


class T2StarExpModel(fitting.Model):
    def __init__(self, pixel_array, te, mask=None, multithread=True):
        """
        A class for fitting T2* data to a mono-exponential model.

        Parameters
        ----------
        pixel_array : np.ndarray
            An array containing the signal from each voxel at each echo
            time with the last dimension being time i.e. the array needed to
            generate a 3D T2* map would have dimensions [x, y, z, TE].
        te : np.ndarray
            An array of the echo times used for the last dimension of the
            pixel_array. In milliseconds.
        mask : np.ndarray, optional
            A boolean mask of the voxels to fit. Should be the shape of the
            desired T2* map rather than the raw data i.e. omit the time
            dimension.
        multithread : bool, optional
            Default True
            If True, the fitting will be performed in parallel using all
            available cores
        """

        super().__init__(pixel_array, te, two_param_eq, mask, multithread)
        self.bounds = ([0, 0], [700, 100])
        self.initial_guess = [20, 1]
        self.generate_lists()


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
    r2 : np.ndarray
        The R-Squared value of the fit, values close to 1 indicate a good
        fit, lower values indicate a poorer fit
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
            or method == '2p_exp', f'method must be loglin or 2p_exp. You ' \
                                   f'entered {method}'
        assert multithread is True \
            or multithread is False \
            or multithread == 'auto', f'multithreaded must be True, False ' \
                                      f'or auto. You entered {multithread}'

        # Normalise the data so its roughly in the same range across vendors
        self.scale = np.nanmax(pixel_array)
        self.pixel_array = pixel_array / self.scale

        self.shape = pixel_array.shape[:-1]
        self.n_te = pixel_array.shape[-1]
        self.n_vox = np.prod(self.shape)
        self.affine = affine

        # Generate a mask if there isn't one specified
        if mask is None:
            self.mask = np.ones(self.shape, dtype=bool)
        else:
            self.mask = mask.astype(bool)

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

        # Initialise an exponential model, even if we're using loglin fit,
        # so we're using the same limits etc
        self._exp_model = T2StarExpModel(self.pixel_array, self.echo_list,
                                         self.mask, self.multithread)
        if self.method == 'loglin':
            popt, error, r2 = self._loglin_fit()
        else:
            popt, error, r2 = fitting.fit_image(self._exp_model)

        self.t2star_map = popt[0]
        self.m0_map = popt[1]
        self.t2star_err = error[0]
        self.m0_err = error[1]
        self.r2 = r2

        # Filter values that are very close to models upper bounds of T2* or
        # M0 out.
        threshold = 0.999  # 99.9% of the upper bound
        bounds_mask = ((self.t2star_map >
                        self._exp_model.bounds[1][0] * threshold) |
                       (self.m0_map > self._exp_model.bounds[1][1] * threshold))
        self.t2star_map[bounds_mask] = 0
        self.m0_map[bounds_mask] = 0
        self.t2star_err[bounds_mask] = 0
        self.m0_err[bounds_mask] = 0
        self.r2[bounds_mask] = 0

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

        # Scale the data back to the original range
        self.m0_map *= self.scale
        self.m0_err *= self.scale

    def _loglin_fit(self):
        if self.multithread:
            with ProcessPool() as executor:
                results = executor.map(self._fit_loglin_signal,
                                       self._exp_model.signal_list,
                                       self._exp_model.x_list,
                                       self._exp_model.mask_list,
                                       [self._exp_model] * self.n_vox)
        else:
            results = list(tqdm(map(self._fit_loglin_signal,
                                    self._exp_model.signal_list,
                                    self._exp_model.x_list,
                                    self._exp_model.mask_list,
                                    [self._exp_model] * self.n_vox),
                                total=self.n_vox))
        popt_array = np.array([result[0] for result in results])
        popt_list = [popt_array[:, p].reshape(self._exp_model.map_shape) for p
                     in range(self._exp_model.n_params)]
        error_array = np.array([result[1] for result in results])
        error_list = [error_array[:, p].reshape(self._exp_model.map_shape)
                      for p in range(self._exp_model.n_params)]
        r2 = np.array([result[2] for result in results]).reshape(
            self._exp_model.map_shape)
        return popt_list, error_list, r2

    @staticmethod
    def _fit_loglin_signal(sig, te, mask, model):
        if mask is True:
            with np.errstate(divide='ignore', invalid='ignore'):
                sig = np.array(sig)
                te = np.array(te)
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
                        if t2star < 0 or t2star > model.bounds[1][0] or \
                           np.isnan(t2star):
                            t2star = 0
                            m0 = 0
                    else:
                        t2star = 0
                        m0 = 0
                else:
                    t2star = 0
                    m0 = 0
        else:
            t2star = 0
            m0 = 0

        fit_sig = two_param_eq(te, t2star, m0)
        r2 = r2_score(sig, fit_sig)
        t2star_err = np.nan
        m0_err = np.nan
        return (t2star, m0), (t2star_err, m0_err), r2

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
            by the function with R2* measured in ms^-1.
        """
        with np.errstate(divide='ignore'):
            r2star = np.nan_to_num(np.reciprocal(self.t2star_map),
                                   posinf=0, neginf=0)
        return r2star

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
            "m0_err", "r2star", "r2", "mask"].
        """
        os.makedirs(output_directory, exist_ok=True)
        base_path = os.path.join(output_directory, base_file_name)
        if maps == 'all' or maps == ['all']:
            maps = ['t2star', 'm0', 'r2star', 'r2', 'mask']
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
                elif result == 'r2' or result == 'r2_map':
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
                             '"["t2star", "t2star_err", "m0", "m0_err", '
                             '"r2star", "mask"]".')

        return

    def get_fit_signal(self):
        """
        Get the fit signal from the model used to fit the data i.e. the
        simulated signal at each echo time given the estimated T2* and M0.

        Returns
        -------
        fit_signal : np.ndarray
            An array containing the fit signal generated by the model
        """
        fit_signal = np.zeros((self.n_vox, self.n_te))
        params = np.array([self.t2star_map.reshape(-1),
                           self.m0_map.reshape(-1)])

        for n in range(self.n_vox):
            fit_signal[n, :] = two_param_eq(self.echo_list,
                                            params[0, n],
                                            params[1, n])
        fit_signal = fit_signal.reshape((*self.shape, self.n_te))
        return fit_signal


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
    with np.errstate(divide='ignore'):
        signal = m0 * np.exp(-t / t2star)
    return signal
