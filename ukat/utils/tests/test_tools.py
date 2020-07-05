import numpy as np
import numpy.testing as npt
from methods.tools import mask_slices


def test_mask_slices():
    shape = (2, 2, 3)

    # Create mask where all pixels from all slices are True
    full_mask = np.full(shape, True)

    # Create mask where only the pixels from slice index 1 are True
    one_slice_mask = np.full(shape, False)
    one_slice_mask[:, :, 1] = True

    # Create mask where only the pixels from slices index 1 and 2 are True
    two_slice_mask = np.full(shape, False)
    two_slice_mask[:, :, 1] = True
    two_slice_mask[:, :, 2] = True

    # -----------------------
    # Input combination tests
    # -----------------------

    # #1: shape + single slice (int)
    final_mask = mask_slices(shape, 1)
    npt.assert_array_equal(final_mask, one_slice_mask)

    # #2: shape + multiple slices (list)
    final_mask = mask_slices(shape, [1, 2])
    npt.assert_array_equal(final_mask, two_slice_mask)

    # #3: shape + single slice + mask
    final_mask = mask_slices(shape, 1, full_mask)
    npt.assert_array_equal(final_mask, one_slice_mask)

    # #4: shape + multiple slices + mask
    final_mask = mask_slices(shape, [1, 2], one_slice_mask)
    npt.assert_array_equal(final_mask, one_slice_mask)

    # -----------------
    # Error check tests
    # -----------------

    # Wrong type `shape`
    non_tuple_shape = [2, 2, 3]
    npt.assert_raises(ValueError, mask_slices, non_tuple_shape, 1)

    # Wrong type `slices`
    npt.assert_raises(ValueError, mask_slices, shape, np.array([1, 2]))

    # Wrong type mask
    wrong_type_mask = "not a numpy array"
    npt.assert_raises(AssertionError, mask_slices, shape, 1, wrong_type_mask)

    # Wrong dtype mask
    wrong_dtype_mask = np.full(shape, 2)
    npt.assert_raises(AssertionError, mask_slices, shape, 1, wrong_dtype_mask)

    # Not all elements of slices are `ints`
    npt.assert_raises(ValueError, mask_slices, shape, [1, 2.2])
    npt.assert_raises(ValueError, mask_slices, shape, 2.2)

    # Slices out of range tests
    npt.assert_raises(ValueError, mask_slices, shape, [0, 3])
    npt.assert_raises(ValueError, mask_slices, shape, 3)

    # `mask` with different dimensions than `shape`
    mismatched_mask = np.full((2, 3, 3), True)
    npt.assert_raises(AssertionError, mask_slices, shape, 1, mismatched_mask)
