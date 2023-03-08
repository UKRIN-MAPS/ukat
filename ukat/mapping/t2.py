import os
import nibabel as nib
import numpy as np
import concurrent.futures

from . import fitting
from itertools import compress
from tqdm import tqdm
from scipy.optimize import curve_fit


class T2Model:
    def __init__(self, pixel_array, te, method='2p_exp', mask=None,
                 multithread=True):
        self.pixel_array = pixel_array
        self.map_shape = pixel_array.shape[:-1]
        self.x = te
        self.method = method
        self.mask = mask
        self.multithread = multithread
        self.n_x = pixel_array.shape[-1]

        if method == '2p_exp':
            self.eq = two_param_eq
            self.n_params = 2
            self.bounds = ([0, 0], [1000, 100000000])
            self.initial_guess = [20, 10000]
        elif self.method == '3p_exp':
            self.eq = three_param_eq
            self.n_params = 3
            self.bounds = ([0, 0, 0], [1000, 100000000, 1000000])
            self.initial_guess = [20, 10000, 500]

        self.signal_list = self.pixel_array.reshape(-1, self.n_x).tolist()
        self.x_list = [self.x] * len(self.signal_list)
        self.p0_list = [self.initial_guess] * len(self.signal_list)

        if self.mask is None:
            self.mask_list = [True] * len(self.signal_list)
        else:
            self.mask_list = self.mask.reshape(-1).tolist()

    def threshold_noise(self, threshold=0):
        for ind, (sig, te, p0) in enumerate(zip(self.signal_list,
                                                self.x_list,
                                                self.p0_list)):
            self.signal_list[ind] = np.array(list(compress(sig, np.array(sig) >
                                                  threshold)))
            self.x_list[ind] = np.array(list(compress(te, np.array(sig) >
                                             threshold)))
            self.p0_list[ind] = np.array(list(compress(p0, np.array(sig) >
                                              threshold)))


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
               or multithread == 'auto', f'multithreaded must be True,' \
                                         f'False or auto. You entered ' \
                                         f'{multithread}'

        if method != '2p_exp' and method != '3p_exp':
            raise ValueError(f'method can be 2p_exp or 3p_exp only. You '
                             f'specified {method}')

        self.pixel_array = pixel_array
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
        fitting_model = T2Model(self.pixel_array, self.echo_list,
                                self.method, self.mask, self.multithread)

        if self.noise_threshold > 0:
            fitting_model.threshold_noise(self.noise_threshold)
        popt, error, r2 = fitting.fit_image(fitting_model)
        self.t2_map = popt[0]
        self.m0_map = popt[1]
        self.t2_err = error[0]
        self.m0_err = error[1]
        self.r2 = r2

        if self.method == '3p_exp':
            self.b_map = popt[2]
            self.b_err = error[2]

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
                    mask_nifti = nib.Nifti1Image(self.mask.astype(np.uint16),
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
    with np.errstate(divide='ignore'):
        signal = m0 * np.exp(-t / t2)
    return signal


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
    with np.errstate(divide='ignore'):
        signal = m0 * np.exp(-t / t2) + b
    return signal
