import pytest
import numpy as np
import numpy.testing as npt
import ukat.utils.tools as tools
from ukat.utils import arraystats


class TestConvertToPiRange:

    # Gold Standard = [mean, std, minimum, maximum]
    # Input: {np.arange(12).reshape((2, 2, 3)) - 6 * np.ones((2, 2, 3))}
    gold_standard = [-7.401486830834377e-17, 1.9718077939258474,
                     -3.141592653589793, 3.1415926535897922]

    # Create arrays for testing
    array = np.arange(12).reshape((2, 2, 3)) - 6 * np.ones((2, 2, 3))
    array_positive = np.arange(12).reshape((2, 2, 3))
    array_negative = -np.arange(12).reshape((2, 2, 3))
    array_pi_range = array / 2

    def test_pi_range_result(self):
        pi_range_calculated = tools.convert_to_pi_range(self.array)
        stats = arraystats.ArrayStats(pi_range_calculated).calculate()
        npt.assert_allclose([stats["mean"]["3D"], stats["std"]["3D"],
                                    stats["min"]["3D"], stats["max"]["3D"]],
                                    self.gold_standard, rtol=1e-6, atol=1e-4)

    def test_if_ranges(self):
        # Test for values > 3.2
        result_over = tools.convert_to_pi_range(self.array_positive)
        assert np.amax(result_over) < np.amax(self.array_positive)
        # Test for values < -3.2
        result_under = tools.convert_to_pi_range(self.array_negative)
        assert np.amin(result_under) > np.amin(self.array_negative)
        # Test for values > 3.2 and < -3.2
        result_under_over = tools.convert_to_pi_range(self.array)
        assert np.amax(result_under_over) < np.amax(self.array)
        assert np.amin(result_under_over) > np.amin(self.array)

    def test_else_range(self):
        result = tools.convert_to_pi_range(self.array_pi_range)
        assert (result == self.array_pi_range).all()

    def test_input_array_type_assertion(self):
        # Empty array
        with pytest.raises(ValueError):
            tools.convert_to_pi_range(np.array([]))
        # No input argument
        with pytest.raises(TypeError):
            tools.convert_to_pi_range(None)
        # String
        with pytest.raises(TypeError):
            tools.convert_to_pi_range("abcdef")


class TestResizeArray:

    # Create arrays for testing
    array_2d = np.arange(100).reshape((10, 10))
    array_3d = np.arange(500).reshape((10, 10, 5))
    array_4d = np.arange(5000).reshape((10, 10, 5, 10))

    def test_no_resize(self):
        resized_array_2d = tools.resize_array(self.array_2d)
        resized_array_3d = tools.resize_array(self.array_3d, factor=1)
        resized_array_4d = tools.resize_array(self.array_4d, target_size=10)
        assert (resized_array_2d == self.array_2d).all()
        assert (resized_array_3d == self.array_3d).all()
        assert (resized_array_4d == self.array_4d).all()

    def test_output_shapes(self):
        resized_array_1 = tools.resize_array(self.array_3d, factor=2)
        resized_array_2 = tools.resize_array(self.array_4d, target_size=20)
        assert np.shape(resized_array_1)[0] != 10
        assert np.shape(resized_array_1)[0] == 20
        assert np.shape(resized_array_1)[0] == np.shape(resized_array_1)[1]
        assert np.shape(resized_array_2)[0] != 10
        assert np.shape(resized_array_2)[0] == 20
        assert np.shape(resized_array_1)[0] == np.shape(resized_array_2)[0]
        assert np.shape(resized_array_1)[1] == np.shape(resized_array_2)[1]
        assert np.shape(resized_array_1)[2] != 10
        assert np.shape(resized_array_1)[2] == np.shape(self.array_3d)[2]
        assert np.shape(resized_array_2)[3] != 20
        assert np.shape(resized_array_2)[3] == np.shape(self.array_4d)[3]

    def test_input_array_type_assertion(self):
        # Empty array
        with pytest.raises(IndexError):
            tools.resize_array(np.array([]))
        # No input argument
        with pytest.raises(IndexError):
            tools.resize_array(None)
        # String
        with pytest.raises(IndexError):
            tools.resize_array("abcdef")


class TestMaskSlices:

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

    def test_single_slice(self):
        # #1: shape + single slice (int)
        final_mask = tools.mask_slices(self.shape, 1)
        assert (final_mask == self.one_slice_mask).all()

    def test_multiple_slices(self):
        # #2: shape + multiple slices (list)
        final_mask = tools.mask_slices(self.shape, [1, 2])
        assert (final_mask == self.two_slice_mask).all()

    def test_masked_single_slice(self):
        # #3: shape + single slice + mask
        final_mask = tools.mask_slices(self.shape, 1, self.full_mask)
        assert (final_mask == self.one_slice_mask).all()

    def test_masked_masked_slices(self):
        # #4: shape + multiple slices + mask
        final_mask = tools.mask_slices(self.shape, [1, 2], self.full_mask)
        assert (final_mask == self.two_slice_mask).all()

    def test_shape_assertion(self):
        # Wrong type `shape`
        non_tuple_shape = [2, 2, 3]
        with pytest.raises(ValueError):
            tools.mask_slices(non_tuple_shape, 1)
        # `mask` with different dimensions than `shape`
        mismatched_mask = np.full((2, 3, 3), True)
        with pytest.raises(AssertionError):
            tools.mask_slices(self.shape, 1, mismatched_mask)

    def test_slices_type_assertion(self):
        # Wrong type `slices`
        with pytest.raises(ValueError):
            tools.mask_slices(self.shape, np.array([1, 2]))
        # Not all elements of slices are `ints`
        with pytest.raises(ValueError):
            tools.mask_slices(self.shape, [1, 2.2])
        with pytest.raises(ValueError):
            tools.mask_slices(self.shape, 2.2)

    def test_slice_ranges_assertion(self):
        # Slices out of range tests
        with pytest.raises(ValueError):
            tools.mask_slices(self.shape, [0, 3])
        with pytest.raises(ValueError):
            tools.mask_slices(self.shape, 3)

    def test_mask_type_assertion(self):
        # Wrong dtype mask
        wrong_type_mask = "not a numpy array"
        with pytest.raises(AssertionError):
            tools.mask_slices(self.shape, 1, wrong_type_mask)

    def test_mask_dtype_assertion(self):
        # Wrong dtype mask
        wrong_dtype_mask = np.full(self.shape, 2)
        with pytest.raises(AssertionError):
            tools.mask_slices(self.shape, 1, wrong_dtype_mask)
