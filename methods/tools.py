"""
This module contains multiple auxiliary functions that might be used by
multiple algorithms

"""
import numpy as np
from skimage.transform import resize
from skimage.restoration import unwrap_phase


def unwrap_phase_image(pixel_array):
    # Wrapping of the phase image according to
    # https://scikit-image.org/docs/dev/auto_examples/filters/plot_phase_unwrap.html
    wrapped_phase = np.angle(np.exp(2j * pixel_array))
    return unwrap_phase(wrapped_phase)


def convert_to_radians(pixel_array):
    """Set the image units to radians"""
    if (np.amax(pixel_array) > 3.2) and (np.amin(pixel_array) < -3.2):
        # The value 3.2 was chosen instead of np.pi in order
        # to give some margin.
        pi_array = np.pi * np.ones(np.shape(pixel_array))
        min_array = np.amin(pixel_array) * np.ones(np.shape(pixel_array))
        max_array = np.amax(pixel_array) * np.ones(np.shape(pixel_array))
        radians_array = (2.0 * pi_array * (pixel_array - min_array) /
                         (max_array - min_array)) - pi_array
    else:
        # It means it's already in radians
        radians_array = pixel_array
    return radians_array


def convert_to_pi_range(pixel_array):
    """Set the image values to the interval [-pi, pi]"""
    pi_array = np.pi * np.ones(np.shape(pixel_array))
    return ((pixel_array + pi_array) % (2 * pi_array)) - pi_array


def resize_array(pixel_array, pixel_spacing, reconst_pixel=None):
    """Resizes the given pixel_array, using reconst_pixel as the reference
        of the resizing operation. This method assumes that the dimension
        axes=0 of pixel_array is the number of slices for np.shape > 2."""
    # This is so that data shares the same resolution and size

    if reconst_pixel is not None:
        fraction = reconst_pixel / pixel_spacing
    else:
        fraction = 1

    if len(np.shape(pixel_array)) == 2:
        pixel_array = resize(
            pixel_array, (pixel_array.shape[0] // fraction,
                          pixel_array.shape[1] // fraction),
            anti_aliasing=True)
    elif len(np.shape(pixel_array)) == 3:
        pixel_array = resize(pixel_array, (pixel_array.shape[0],
                                           pixel_array.shape[1] // fraction,
                                           pixel_array.shape[2] // fraction),
                             anti_aliasing=True)
    elif len(np.shape(pixel_array)) == 4:
        pixel_array = resize(pixel_array, (pixel_array.shape[0],
                                           pixel_array.shape[1],
                                           pixel_array.shape[2] // fraction,
                                           pixel_array.shape[3] // fraction),
                             anti_aliasing=True)
    elif len(np.shape(pixel_array)) == 5:
        pixel_array = resize(pixel_array, (pixel_array.shape[0],
                                           pixel_array.shape[1],
                                           pixel_array.shape[2],
                                           pixel_array.shape[3] // fraction,
                                           pixel_array.shape[4] // fraction),
                             anti_aliasing=True)
    elif len(np.shape(pixel_array)) == 6:
        pixel_array = resize(pixel_array, (pixel_array.shape[0],
                                           pixel_array.shape[1],
                                           pixel_array.shape[2],
                                           pixel_array.shape[3],
                                           pixel_array.shape[4] // fraction,
                                           pixel_array.shape[5] // fraction),
                             anti_aliasing=True)

    return pixel_array


def mask_slices(shape, slices, mask=None):
    """Get mask to limit processing to specific slices

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
