"""
This module contains multiple auxiliary functions that might be used by
multiple algorithms

"""
import numpy as np
from scipy.ndimage import zoom


def convert_to_pi_range(pixel_array):
    """
    Rescale the image values to the interval [-pi, pi].

    Parameters
    ----------
    pixel_array : np.ndarray

    Returns
    -------
    radians_array : np.ndarray
        An array containing with the same shape as pixel_array
        scaled to the range [-pi, pi].
    """
    if (np.amax(pixel_array) > np.pi) or (np.amin(pixel_array) < -np.pi):
        pi_array = np.pi * np.ones(np.shape(pixel_array))
        min_array = np.amin(pixel_array) * np.ones(np.shape(pixel_array))
        max_array = np.amax(pixel_array) * np.ones(np.shape(pixel_array))
        radians_array = (2.0 * pi_array * (pixel_array - min_array) /
                         (max_array - min_array)) - pi_array
    else:
        # It means it's already on the interval [-pi, pi]
        radians_array = pixel_array
    return radians_array


def resize_array(pixel_array, factor=1, target_size=None):
    """
    Resizes the given pixel_array, using target_size as the reference
    of the resizing operation (if None, then there's no resizing).
    This method applies a resize_factor to the first 2 axes of the input array.
    The remaining axes are left unchanged.
    Example 1: (10, 10, 2) => (5, 5, 2) with resize_factor = 0.5
    Example 2: (10, 10, 10, 2) => (20, 20, 10, 2) with resize_factor = 2

    Parameters
    ----------
    pixel_array : np.ndarray

    factor : boolean
        Optional input argument. This is the resize factor defined by the user
        and it is applied in the scipy.ndimage.zoom

    target_size : boolean
        Optional input argument. By default, this script does not apply the
        scipy wrap_around in the input image.

    Returns
    -------
    resized_array : np.ndarray where the size of the first 2 dimensions
        is np.shape(pixel_array) * factor. The remaining dimensions (or axes)
        will have a the same size as in pixel_array.
    """
    if target_size is not None:
        factor = target_size / np.shape(pixel_array)[0]

    resize_factor = np.ones(len(np.shape(pixel_array)))
    resize_factor[0] = factor
    resize_factor[1] = factor
    resized_array = zoom(pixel_array, resize_factor)

    return resized_array


def mask_slices(shape, slices, mask=None):
    """
    Get mask to limit processing to specific slices.

    This function allows to quickly get a mask of Trues of the right shape to
    limit processing to specific slices. If `mask` is provided it outputs a new
    mask corresponding to the input mask but only on the specified slices.

    Parameters
    ----------
    shape : tuple
        shape of mask to be created
    slices : int or list (of ints)
        slice indices where mask is to be True
    mask : np.ndarray (of booleans)
        original mask, if provided this function will return a mask of Falses
        in all elements except at the locations of the True elements of this
        `mask` in the slices given by `slices`

    Returns
    -------
    np.ndarray (of booleans)
    """
    # Input type checks
    if not isinstance(shape, tuple):
        raise ValueError("`shape` must be a tuple")

    if not isinstance(slices, (int, list)):
        raise ValueError("`slices` must be an integer or a list of integers")

    # Check elements of `slices` are ints
    if (isinstance(slices, list) and
       not all(isinstance(x, int) for x in slices)):
        raise ValueError("Every element of `slices` must be an integer")

    # Check elements of `slices` within range defined by `shape`
    if isinstance(slices, int):
        s_min = slices
        s_max = slices
    elif isinstance(slices, list):
        s_min = min(slices)
        s_max = max(slices)

    if not(s_min >= 0 and s_max+1 <= shape[2]):
        msg = f"The elements of `slices` must be > 0 and <= {shape[2]-1}"
        raise ValueError(msg)

    # Ensure shape and the dimensions of mask match
    if mask is not None:
        assert isinstance(mask, np.ndarray), "`mask` must be a numpy"
        assert mask.dtype == "bool", "The elements of `mask` must be bool"
        assert (shape == mask.shape), "The shape of `mask` must match `shape`"

    # Make mask of Trues at the specified slices
    template = np.full(shape, False)
    template[:, :, slices] = True

    # If `mask` provided, apply it to the template
    if mask is None:
        final_mask = template
    else:
        final_mask = np.logical_and(mask, template)

    return final_mask
