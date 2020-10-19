import numpy as np
from skimage.restoration import unwrap_phase
from ukat.utils.tools import convert_to_pi_range


def b0map(pixel_array, echo_list, unwrap=True, wrap_around=False):
    """
    Generates a B0 map from a series of volumes collected
    with 2 different echo times. It can also generate B0 map
    from a phase_difference image.

    Parameters
    ----------
    pixel_array : 4D/3D array
        A 4D/3D array containing the phase image at each
        echo time i.e. the dimensions of the array are
        [x, y, TE] or [x, y, z, TE].
    echo_list : list()
        An array of the echo times in ms used for the last dimension of the
        raw data.
    unwrap : boolean
        Optional input argument. By default, this script applies the
        scipy phase unwrapping for each phase echo image.
    wrap_around : boolean
        Optional input argument. By default, this script does not apply the
        scipy wrap_around in the phase unwrapping for each phase echo image.

    Returns
    -------
    b0 : 2D/3D array
        Array containing the B0 map unwrapped (rad/s)
        If pixel_array is 4D, then B0 map will be 3D.
        If pixel_array is 3D, then B0 map will be 2D.
    """
    # B0 Map accepts inputs with more than 2 echo times
    # Extract each phase image and rescale to [-pi, pi] if not in that range.
    phase0 = convert_to_pi_range(np.squeeze(pixel_array[..., 0]))
    phase1 = convert_to_pi_range(np.squeeze(pixel_array[..., 1]))
    if unwrap:
        # Unwrap each phase image
        phase0 = unwrap_phase(phase0, wrap_around=wrap_around)
        phase1 = unwrap_phase(phase1, wrap_around=wrap_around)
    phase_diff = phase1 - phase0
    delta_te = np.absolute(echo_list[1] - echo_list[0]) * 0.001
    # B0 Map calculation
    b0 = phase_diff / (2 * np.pi * delta_te)
    del phase_diff, delta_te, phase0, phase1
    return b0
