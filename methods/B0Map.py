import numpy as np
from methods.tools import unwrap_phase_image, convert_to_pi_range


def B0Map_unwrap_phase(pixel_array, echo_list):
    """
    Generates a B0 map from a series of volumes collected
    with 2 different echo times. It can also generate B0 map
    from a phase_difference image. It applies
    the scipy unwrapping method accordingly.

    Parameters
    ----------
    pixel_array : 4D/3D array
        A 4D/3D array containing the phase image at each
        echo time i.e. the dimensions of the array are
        [x, y, TE] or [x, y, z, TE]. This array can also be
        a phase_difference, however the length of echo_list
        must be one or none (see below).
    echo_list : list()
        An array of the echo times in ms used for the last dimension of the
        raw data.

    Returns
    -------
    b0 : 2D/3D array
        Array containing the B0 map unwrapped generated by the method (rad/s)
        If pixel_array is 4D, then B0 map will be 3D.
        If pixel_array is 3D, then B0 map will be 2D.

    See the following links regarding the unwrapping approach
    https://scikit-image.org/docs/dev/auto_examples/filters/plot_phase_unwrap.html
    """

    try:
        # Convert to [-pi, pi] if not in that range already
        radians_array = convert_to_pi_range(pixel_array)
        # Is the given array already a Phase Difference or not?
        if len(echo_list) > 1:
            # Unwrap each phase image
            phase0 = unwrap_phase_image(np.squeeze(radians_array[..., 0]))
            phase1 = unwrap_phase_image(np.squeeze(radians_array[..., 1]))
            phase_diff = phase1 - phase0
            deltaTE = np.absolute(echo_list[1] - echo_list[0]) * 0.001
            # This function could be modified to return multiple variables
            # in the future: eg. return phase0, phase1, phase_diff, ...
            del phase0, phase1
        else:
            # If it's a Phase Difference image, it just unwraps the image
            phase_diff = unwrap_phase_image(radians_array)
            # Sometimes, there is no EchoTime tag <=> No echo values parsed
            try:
                deltaTE = echo_list[0] * 0.001
            except Exception as e:
                # If EchoTime is 0 or empty
                deltaTE = 0.001
        # B0 Map calculation
        b0 = phase_diff / (2 * np.pi * deltaTE)
        del phase_diff, deltaTE
        return b0
    except Exception as e:
        print('Error in function B0Map.B0Map_unwrap_phase: ' + str(e))


def B0Map(pixel_array, echo_list):
    """
    Generates a B0 map from a series of volumes collected
    with 2 different echo times. It can also generate B0 map
    from a phase_difference image.

    Parameters
    ----------
    pixel_array : 4D/3D array
        A 4D/3D array containing the phase image at each
        echo time i.e. the dimensions of the array are
        [x, y, TE] or [x, y, z, TE]. This array can also be
        a phase_difference, however the length of echo_list
        must be one or none (see below).
    echo_list : list()
        An array of the echo times in ms used for the last dimension of the
        raw data.

    Returns
    -------
    b0 : 2D/3D array
        Array containing the B0 map unwrapped (rad/s)
        If pixel_array is 4D, then B0 map will be 3D.
        If pixel_array is 3D, then B0 map will be 2D.
    """

    try:
        # Convert to [-pi, pi] if not in that range already
        radians_array = convert_to_pi_range(pixel_array)
        # Is the given array already a Phase Difference or not?
        if len(echo_list) > 1:
            # Extract each phase image
            phase0 = np.squeeze(radians_array[..., 0])
            phase1 = np.squeeze(radians_array[..., 1])
            phase_diff = phase1 - phase0
            deltaTE = np.absolute(echo_list[1] - echo_list[0]) * 0.001
            # This function could be modified to return multiple variables
            # in the future: eg. return phase0, phase1, phase_diff, ...
            del phase0, phase1
        else:
            # If it's a Phase Difference image
            phase_diff = radians_array
            # Sometimes, there is no EchoTime tag <=> No echo values parsed
            try:
                deltaTE = echo_list[0] * 0.001
            except Exception as e:
                # If EchoTime is 0 or empty
                deltaTE = 0.001
        # B0 Map calculation
        b0 = phase_diff / (2 * np.pi * deltaTE)
        del phase_diff, deltaTE
        return b0
    except Exception as e:
        print('Error in function B0Map.B0Map: ' + str(e))
