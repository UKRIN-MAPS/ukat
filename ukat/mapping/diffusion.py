"""
Diffusion imaging module

"""
import os
import nibabel as nib
import numpy as np
import warnings

from dipy.core.gradients import gradient_table, unique_bvals_tolerance
from dipy.reconst.dti import TensorModel
from tqdm import tqdm


def make_gradient_scheme(bvals, bvecs, normalize=True, one_bzero=True):
    """Make gradient scheme from list of bvals and bvecs

    Parameters
    ----------
    bvals : list
        b-values in s/mm2
    bvecs : list of lists
        bvectors (e.g. [[0, 0, 1], [1, 0, 0]])
    normalize : bool, optional (default True)
        Rescales bvecs to have unit length
    one_bzero : bool, optional (default True)
        Ensures gradient scheme only includes one b=0 measurement
        If this is True and bvals does not contain any b=0, a b=0 measurement
        will be included at the start of the acquisition

    Returns
    -------
    string
        gradient scheme with one line per measurement/volume as follows:
        bvec1_x   bvec1_y   bvec1_z   bval1
        bvec2_x   bvec2_y   bvec2_z   bval2
        ...
        bvecN_x   bvecN_y   bvecN_z   bvalN

    Notes
    -----
    This function was created to generate a diffusion scheme for the UKRIN-MAPS
    multishell acquisition where all the nonzero b-values have the same number
    of directions. Currently this does not provide features to generate
    schemes where different shells have different numbers of directions.
    This gradient scheme format was decided with the following in mind:
        1) can be easily written to a text file to allow modifications to it
           to be done without coding
        2) not vendor specific
        3) could be useful as a starting point to convert these schemes to
           vendor-specific formats

    """
    if 0 not in bvals and one_bzero:
        bvals.insert(0, 0)

    if normalize:
        # Rescale bvecs to have norm 1
        bvecs = [v/np.linalg.norm(v) for v in bvecs]

    bvecs = [np.round(x, 8) for x in bvecs]

    # Make gradient scheme
    bzero_counter = 0
    gradient_scheme = ""
    for bvec in bvecs:
        for bval in bvals:
            if bval == 0 and one_bzero and bzero_counter > 0:
                continue
            else:
                gradient_scheme = (f"{gradient_scheme}"
                                   f"{str(bvec[0]).rjust(11)}  "
                                   f"{str(bvec[1]).rjust(11)}  "
                                   f"{str(bvec[2]).rjust(11)}  "
                                   f"{str(bval).rjust(5)}\n")
                if bval == 0:
                    bzero_counter += 1

    # Remove last newline
    if gradient_scheme[-1] == '\n':
        gradient_scheme = gradient_scheme[:-1:]

    return gradient_scheme


class ADC:
    """
    Attributes
    ----------
    adc : np.ndarray
        The estimated ADC in mm^2/s.
    adc_err : np.ndarray
        The certainty in the fit of `adc` in mm^2/s.
    shape : tuple
        The shape of the ADC map.
    n_vox : int
        The number of voxels in the map.
    bvals : 1d numpy array
        All b-values that will be used to generate the maps in s/mm^2.
    u_bvals : 1d numpy array
        The unique b-values used in the experiment e.g. if the experiment
        acquires a single b-0 volume and 64 volumes with b=600 s/mm^2 in
        different directions, u_bvals will be [0, 600].
    n_bvals : int
        The number of unique b-values acquired in the experiment.
    n_grad : int
        Total number of diffusion values/vectors acquired e.g. if the
        experiment acquires six directions at 10 gradient strengths and a
        b-0 volume, n_grad will be 61.
    pixel_array_mean : np.ndarray
        The average of the `pixel_array`across bvecs e.g. if `pixel_array`
        contains six volumes acquired with b=600 s/mm^2 in different
        directions, these six volumes will be averaged together.
    """
    def __init__(self, pixel_array, affine, bvals, mask=None, ukrin_b=False):
        """Initialise a ADC class instance.

        Parameters
        ----------
        pixel_array : (..., N) np.ndarray
            A array containing the signal from each voxel at each
            diffusion sensitising parameter. The final dimension should be
            different diffusion weightings/directions.
        affine : np.ndarray
            A matrix giving the relationship between voxel coordinates and
            world coordinates.
        bvals : (N,) np.array
            An array of the b-values used for the last dimension of the raw
            data. In s/mm^2.
        mask : np.ndarray, optional
            A boolean mask of the voxels to fit. Should be the shape of the
            desired map rather than the raw data i.e. omit the last dimension.
        ukrin_b : bool, optional
            If True, only b-values of 0, 100, 200 and 800 s/mm^2 will be
            included in the ADC fit. This aligns with Ljimani A, et al.
            Consensus-based technical recommendations for clinical translation
            of renal diffusion-weighted MRI.
            Magn Reson Mater Phy 2020;33:177–195
            doi: 10.1007/s10334-019-00790-y.
            If False, all b-values supplied will be used to fit ADC.
        """
        ukrin_b_test = np.array([0, 100, 200, 800])
        # Sanity checks
        assert (pixel_array.shape[-1]
                == len(bvals)), 'Number of bvals does not match number of ' \
                                'gradients in pixel_array'

        if ukrin_b:
            self.b_mask = np.isin(bvals, ukrin_b_test)
        else:
            self.b_mask = np.full(len(bvals), True, dtype=bool)

        self.pixel_array = pixel_array[..., self.b_mask]
        self.shape = pixel_array.shape[:-1]
        self.n_vox = np.prod(self.shape)
        self.bvals = bvals[self.b_mask]
        self.n_grad = len(self.bvals)
        self.u_bvals = unique_bvals_tolerance(self.bvals, 1)
        self.n_bvals = len(self.u_bvals)
        self.affine = affine
        # Generate a mask if there isn't one specified
        if mask is None:
            self.mask = np.ones(self.shape, dtype=bool)
        else:
            self.mask = mask
            # Don't process any nan values
        self.mask[np.isnan(np.sum(pixel_array, axis=-1))] = False
        self.mask[np.sum(pixel_array <= 0, axis=-1, dtype=bool)] = False
        self.pixel_array = np.nan_to_num(self.pixel_array)

        self.pixel_array_mean = self.__mean_over_directions__()

        self.adc, self.adc_err = self.__fit__()

    def __mean_over_directions__(self):
        """
        Calculates the mean signal across different directions at each unique
        b-value e.g. if `pixel_array` contains six volumes acquired with
        b=600 s/mm^2 in different directions, these six volumes will be
        averaged together.

        Returns
        -------
        pixel_array_mean : np.ndarray
            The average of the `pixel_array` across bvecs
        """
        pixel_array_mean = np.zeros((*self.shape, self.n_bvals))
        for ind, bval in enumerate(self.u_bvals):
            pixel_array_mean[..., ind]\
                = np.mean(self.pixel_array[..., self.bvals == bval], axis=-1)
        return pixel_array_mean

    def __fit__(self):
        # Initialise maps
        adc_map = np.zeros(self.n_vox)
        adc_err = np.zeros(self.n_vox)

        mask = self.mask.flatten()
        signal = self.pixel_array_mean.reshape(-1, self.n_bvals)
        idx = np.argwhere(mask).squeeze()
        with tqdm(total=idx.size) as progress:
            for ind in idx:
                sig = signal[ind, :]
                adc_map[ind], adc_err[ind] = \
                    self.__fit_signal__(sig, self.u_bvals)
                progress.update(1)
        adc_map[adc_map < 0] = 0
        adc_err[adc_map < 0] = 0

        # Reshape results into raw data shape
        adc_map = adc_map.reshape(self.shape)
        adc_err = adc_err.reshape(self.shape)

        return adc_map, adc_err

    @staticmethod
    def __fit_signal__(sig, bvals):
        try:
            popt, pvar = np.polyfit(bvals[sig > 0], np.log(sig[sig > 0]), 1,
                                    cov=True)
            adc = -popt[0]
            adc_err = np.sqrt(pvar[0, 0])
        except np.linalg.LinAlgError:
            adc = 0
            adc_err = 0

        return adc, adc_err

    def to_nifti(self, output_directory=os.getcwd(), base_file_name='Output',
                 maps='all'):
        """Exports maps generated by the ADC class as NIFTI.

        Parameters
        ----------
        output_directory : string, optional
            Path to the folder where the NIFTI files will be saved.
        base_file_name : string, optional
            Filename of the resulting NIFTI. This code appends the extension.
            Eg., base_file_name = 'Output' will result in 'Output.nii.gz'.
        maps : list or 'all', optional
            List of maps to save to NIFTI. This should either the string "all"
            or a list of maps from ["adc", "adc_err", "mask"].
        """
        os.makedirs(output_directory, exist_ok=True)
        base_path = os.path.join(output_directory, base_file_name)
        if maps == 'all' or maps == ['all']:
            maps = ['adc', 'adc_err', 'mask']
        if isinstance(maps, list):
            for result in maps:
                if result == 'adc' or result == 'adc_map':
                    adc_nifti = nib.Nifti1Image(self.adc, affine=self.affine)
                    nib.save(adc_nifti, base_path + '_adc_map.nii.gz')
                elif result == 'adc_err' or result == 'adc_err_map':
                    adc_err_nifti = nib.Nifti1Image(self.adc_err,
                                                    affine=self.affine)
                    nib.save(adc_err_nifti, base_path + '_adc_err.nii.gz')
                elif result == 'mask':
                    mask_nifti = nib.Nifti1Image(self.mask.astype(int),
                                                 affine=self.affine)
                    nib.save(mask_nifti, base_path + '_mask.nii.gz')
        else:
            raise ValueError('No NIFTI file saved. The variable "maps" '
                             'should be "all" or a list of maps from '
                             '"["adc", "adc_err", "mask"]".')


class DTI:
    """
    Attributes
    ----------
    md : np.ndarray
        The estimated mean diffusivity values in mm^2/s.
    fa : np.ndarray
        The estimated fractional anisotropy values.
    color_fa : np.ndarray
        The estimated directional fractional anisotropy represented as red,
        green and blue corresponding to correspond to fractional anisotropy
        in the x, y and z directions respectively.
    shape : tuple
        The shape of the resulting maps
    bvals : 1d numpy array
        All b-values that will be used to generate the maps in s/mm^2.
    u_bvals : 1d numpy array
        The unique b-values used in the experiment e.g. if the experiment
        acquires a single b-0 volume and 64 volumes with b=600 s/mm^2 in
        different directions, u_bvals will be [0, 600].
    n_bvals : int
        The number of unique b-values acquired in the experiment.
    bvecs : (N, 3) numpy array
        All b-vectors that will be used to generate the maps.
    u_bvecs : (M, 3) numpy array
        The unique b-vectors used in the experiment e.g. if the experiment
        acquires six directions at 10 gradient strengths, u_bvecs will be a
        6 x 3 numpy array.
    n_bvecs : int
        The number of unique b-vectors acquired in the experiment.
    n_grad : int
        Total number of diffusion values/vectors acquired e.g. if the
        experiment acquires six directions at 10 gradient strengths and a
        b-0 volume, n_grad will be 61.
    gtab : dipy GradientTable
        The dipy gradient table used to generate maps.
    tensor_fit : dipy TensorModel after fitting
        The fit dipy tensor model, can be used to recall additional parameters.
    """
    def __init__(self, pixel_array, affine, bvals, bvecs, mask=None,
                 ukrin_b=False):
        """Initialise a DTI class instance.

        Parameters
        ----------
        pixel_array : (..., N) np.ndarray
            A array containing the signal from each voxel at each
            diffusion sensitising parameter. The final dimension should be
            different diffusion weightings/directions.
        affine : np.ndarray
            A matrix giving the relationship between voxel coordinates and
            world coordinates.
        bvals : (N,) np.array
            An array of the b-values used for the last dimension of the raw
            data. In s/mm^2.
        bvecs : (N, 3) np.array
            An array of the b-vectors used for the last dimension of the raw
            data. In s/mm^2.
        mask : np.ndarray, optional
            A boolean mask of the voxels to fit. Should be the shape of the
            desired map rather than the raw data i.e. omit the last dimension.
        ukrin_b : bool, optional
            If True, only b-values of 0, 100, 200 and 800 s/mm^2 will be
            included in the fit. This aligns with Ljimani A, et al.
            Consensus-based technical recommendations for clinical translation
            of renal diffusion-weighted MRI.
            Magn Reson Mater Phy 2020;33:177–195
            doi: 10.1007/s10334-019-00790-y.
            If False, all b-values supplied will be used.
        """
        ukrin_b_test = np.array([0, 100, 200, 800])
        # Some sanity checks
        assert (pixel_array.shape[-1]
                == len(bvals)), 'Number of bvals does not match number of ' \
                                'gradients in pixel_array'
        if bvecs.shape[1] != 3 and bvecs.shape[0] == 3:
            bvecs = bvecs.T
            warnings.warn(f'bvecs should be (N, 3). Because your bvecs array '
                          'is {bvecs.shape} it has been transposed to {'
                          'bvecs.T.shape}.')
        assert (bvecs.shape[1] == 3)
        assert (pixel_array.shape[-1] == bvecs.shape[0]), 'Number of bvecs ' \
                                                          'does not match ' \
                                                          'number of ' \
                                                          'gradients in ' \
                                                          'pixel_array'
        if ukrin_b:
            self.b_mask = np.isin(bvals, ukrin_b_test)
        else:
            self.b_mask = np.full(len(bvals), True, dtype=bool)

        self.pixel_array = pixel_array[..., self.b_mask]
        self.shape = self.pixel_array.shape[:-1]
        self.bvals = bvals[self.b_mask]
        self.bvecs = bvecs[self.b_mask, :]
        self.n_grad = len(self.bvals)
        self.u_bvals = unique_bvals_tolerance(self.bvals, 1)
        self.n_bvals = len(self.u_bvals)
        self.u_bvecs = np.unique(self.bvecs, axis=0)
        self.n_bvecs = len(self.u_bvecs)
        self.affine = affine
        self.mask = mask
        self.gtab = gradient_table(self.bvals, self.bvecs, b0_threshold=0)
        tensor_model = TensorModel(self.gtab)
        self.tensor_fit = tensor_model.fit(self.pixel_array, mask=self.mask)
        self.md = self.tensor_fit.md
        self.fa = self.tensor_fit.fa
        self.color_fa = self.tensor_fit.color_fa

    def to_nifti(self, output_directory=os.getcwd(), base_file_name='Output',
                 maps='all'):
        """Exports maps generated by the DTI class as NIFTI.

        Parameters
        ----------
        output_directory : string, optional
            Path to the folder where the NIFTI files will be saved.
        base_file_name : string, optional
            Filename of the resulting NIFTI. This code appends the extension.
            Eg., base_file_name = 'Output' will result in 'Output.nii.gz'.
        maps : list or 'all', optional
            List of maps to save to NIFTI. This should either the string "all"
            or a list of maps from ["md", "fa", "color_fa", "mask"].
        """
        os.makedirs(output_directory, exist_ok=True)
        base_path = os.path.join(output_directory, base_file_name)
        if maps == 'all' or maps == ['all']:
            maps = ['md', 'fa', 'color_fa', 'mask']
        if isinstance(maps, list):
            for result in maps:
                if result == 'md' or result == 'md_map':
                    md_nifti = nib.Nifti1Image(self.md, affine=self.affine)
                    nib.save(md_nifti, base_path + '_md_map.nii.gz')
                elif result == 'fa' or result == 'fa_map':
                    fa_nifti = nib.Nifti1Image(self.fa, affine=self.affine)
                    nib.save(fa_nifti, base_path + '_fa_map.nii.gz')
                elif result == 'color_fa' or result == 'color_fa_map':
                    color_fa_nifti = nib.Nifti1Image(self.color_fa,
                                                     affine=self.affine)
                    nib.save(color_fa_nifti, base_path +
                             '_color_fa_map.nii.gz')
                elif result == 'mask':
                    mask_nifti = nib.Nifti1Image(self.mask.astype(int),
                                                 affine=self.affine)
                    nib.save(mask_nifti, base_path + '_mask.nii.gz')
        else:
            raise ValueError('No NIFTI file saved. The variable "maps" '
                             'should be "all" or a list of maps from '
                             '"["md", "fa", "color_fa", "mask"]".')
