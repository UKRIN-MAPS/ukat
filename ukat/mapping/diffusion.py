"""
Diffusion-weighted imaging module

"""
import os
import warnings
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.reconst.ivim import IvimModel
from dipy.reconst.dti import TensorModel

# Ensure dipy's defaults do not consider low non-zero b-values to be 0
B0_THRESHOLD = 0

# Default direction vectors for powder averaged b=0 and non-b=0 data
BVECS_PA_B0 = np.array([0, 0, 0])
BVECS_PA_NOT_B0 = np.array([0, 0, 1])


def powder_avg(data, bvals):
    """"Powder average DWI time series

    Parameters
    ----------
    data : numpy.ndarray
        4D image array [x, y, z, t]
    bvals : numpy.ndarray
        b-values (N, )

    Returns
    -------
    data_pa : numpy.ndarray
        Powder averaged 4D image array [x, y, z, b]
    bvals_pa : numpy.ndarray
        Powder averaged b-values (N, )
    bvecs_pa : numpy.ndarray
        Powder averaged b-vectors (N, 3)

    Notes
    --------
    The powder averaged b-vectors are defined as constants such that they have
    norm 1 for corresponding non-zero b-values

    """
    # Create powder averaged (pa'd) bvals
    bvals_pa = np.unique(bvals)

    # Pre-allocate data_pa
    data_pa_dims = tuple(np.append(data.shape[:-1], np.size(bvals_pa)))
    data_pa = np.empty(data_pa_dims)
    data_pa.fill(np.nan)

    # Pre-allocate bvecs_pa
    bvecs_pa_dims = (np.size(bvals_pa), 3)
    bvecs_pa = np.empty(bvecs_pa_dims)
    bvecs_pa.fill(np.nan)

    # Powder average data set and generate pa'd bvecs
    for i, bval in enumerate(bvals_pa):

        # Get indices of volumes with current bval
        c_idxs = bvals == bvals_pa[i]

        # Create data_pa
        c_pa = np.mean(data[:, :, :, c_idxs], axis=3)
        data_pa[:, :, :, i] = c_pa

        # Create bvecs_pa
        if bval == 0:
            bvecs_pa[i, :] = BVECS_PA_B0
        else:
            bvecs_pa[i, :] = BVECS_PA_NOT_B0

    if np.isnan(data_pa).any():
        raise RuntimeError("No NaNs should exist at this point")

    return data_pa, bvals_pa, bvecs_pa


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


def ivim(data, bvals, bvecs, mask=None):
    """Compute IVIM maps

    Wrapper for dipy's IvimModel from the reconst.ivim.py module to compute
    IVIM maps.

    Parameters
    ----------
    data : numpy.ndarray
        4D image array [x, y, z, t]
    bvals : numpy.ndarray
        b-values (N, )
    bvecs : numpy.ndarray
        b-values (N, 3)
    mask : numpy.ndarray
        3D image array [x, y, z]

    Returns
    -------
    numpy.ndarray
        S0 map
    numpy.ndarray
        D_star map
    numpy.ndarray
        D map
    numpy.ndarray
        f map

    """
    # Powder average (pa) data
    data_pa, bvals_pa, bvecs_pa = powder_avg(data, bvals)

    # Initialise gradient table
    gt = gradient_table(bvals_pa, bvecs_pa, b0_threshold=B0_THRESHOLD)

    # Fit IVIM model
    ivimmodel = IvimModel(gt, fit_method='trr')
    ivimfit = ivimmodel.fit(data_pa, mask=mask)

    # Return IVIM maps
    S0 = ivimfit.S0_predicted
    D_star = ivimfit.D_star
    D = ivimfit.D
    f = ivimfit.perfusion_fraction

    return S0, D_star, D, f


class DTI:
    """
    Attributes
    ----------
    md : np.ndarray
        The estimated mean diffusivity values in mm^2/s
    fa : np.ndarray
        The estimated fractional anisotropy values
    fa_color : np.ndarray
        The estimated direcitonal fractional anisotropy represented as red,
        green and blue corresponding to correspond to fractional anisotropy
        in the x, y and z directions respectively
    shape : tuple
        The shape of the T2* map
    bvals : 1d numpy array
        All b-values that will be used to generate the maps in s/mm^2
    u_bvals : 1d numpy array
        The unique b-values used in the experiment e.g. if the experiment
        acquires a single b-0 volume and 64 volumes with b=600 s/mm^2 in
        different directions, u_bvals will be [0, 600]
    n_bvals : int
        The number of unique b-values acquired in the experiment
    bvecs : (N, 3) numpy array
        All b-vectors that will be used to generate the maps
    u_bvecs : (M, 3) numpy array
        The unique b-vectors used in the experiment e.g. if the experiment
        acquires six directions at 10 gradient strengths, u_bvecs will be a
        6 x 3 numpy array
    n_bvecs : int
        The number of unique b-vectors acquired in the experiment
    n_grad : 1d numpy array
        Total number of diffusion values/vectors acquired e.g. if the
        experiment acquires six directions at 10 gradient strengths and a
        b-0 volume, n_grad will be 61
    gtab : dipy GradientTable
        The dipy gradient table used to generate maps
    tensor_fit : dipy TensorModel after fitting
        The fit dipy tensor model, can be used to recall additional parameters
    """
    def __init__(self, pixel_array, bvals, bvecs, affine, mask=None):
        """Initialise a DTI class instance.

        Parameters
        ----------
        pixel_array : (..., N) np.ndarray
            A array containing the signal from each voxel at each
            diffusion sensitising parameter. The final dimension should be
            different diffusion weightings/directions
        bvals : (N,) np.array
            An array of the b-values used for the last dimension of the raw
            data. In s/mm^2.
        bvecs : (N, 3) np.array
            An array of the b-vectors used for the last dimension of the raw
            data. In s/mm^2.
        affine : np.ndarray
            A matrix giving the relationship between voxel coordinates and
            world coordinates.
        mask : np.ndarray, optional
            A boolean mask of the voxels to fit. Should be the shape of the
            desired map rather than the raw data i.e. omit the last dimension.
        """
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
        self.pixel_array = pixel_array
        self.shape = pixel_array.shape[:-1]
        self.bvals = bvals
        self.bvecs = bvecs
        self.n_grad = len(self.bvals)
        self.u_bvals = np.unique(self.bvals)
        self.n_bvals = len(self.u_bvals)
        self.u_bvecs = np.unique(self.bvecs)
        self.n_bvecs = len(self.u_bvecs)
        self.affine = affine
        self.mask = mask
        self.gtab = gradient_table(bvals, bvecs, b0_threshold=0)
        tensor_model = TensorModel(self.gtab)
        self.tensor_fit = tensor_model.fit(pixel_array, mask=self.mask)
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
