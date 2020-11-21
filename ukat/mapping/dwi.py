"""
Diffusion-weighted imaging module

"""
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst.ivim import IvimModel

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
