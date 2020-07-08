import numpy as np
from ukat.utils.tools import unwrap_phase_image, convert_to_pi_range


def b0map(pixel_array, echo_list, unwrap=True):
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

    Returns
    -------
    b0 : 2D/3D array
        Array containing the B0 map unwrapped (rad/s)
        If pixel_array is 4D, then B0 map will be 3D.
        If pixel_array is 3D, then B0 map will be 2D.
    """

    # Rescale to [-pi, pi] if not in that range already
    radians_array = convert_to_pi_range(pixel_array)
    if unwrap:
        # Extract and unwrap each phase image
        phase0 = unwrap_phase_image(np.squeeze(radians_array[..., 0]))
        phase1 = unwrap_phase_image(np.squeeze(radians_array[..., 1]))
    else:
        # Extract each phase image
        phase0 = np.squeeze(radians_array[..., 0])
        phase1 = np.squeeze(radians_array[..., 1])
    phase_diff = phase1 - phase0
    delta_te = np.absolute(echo_list[1] - echo_list[0]) * 0.001
    # B0 Map calculation
    b0 = phase_diff / (2 * np.pi * delta_te)
    del phase_diff, delta_te, phase0, phase1
    return b0
