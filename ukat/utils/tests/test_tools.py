import pytest
import numpy as np
import ukat.utils.tools as tools


class TestUnwrapPhaseImage:

    # Gold Standard = [mean, std, minimum, maximum]
    # Input: {np.arange(-5, 6)}
    gold_standard = [12.56637061435917, 6.324555320336759,
                     2.566370614359173, 22.566370614359172]

    # Create array for testing
    array = np.arange(-5, 6)

    def test_unwrap_result(self):
        unwrap_calculated = tools.unwrap_phase_image(self.array)
        print(tools.image_stats(unwrap_calculated))
        np.testing.assert_allclose(tools.image_stats(unwrap_calculated),
                                   self.gold_standard, rtol=1e-7, atol=1e-9)

    def test_unwrap_phase_quality(self):
        # Input array different than the output array
        result = tools.unwrap_phase_image(self.array)
        assert (result != self.array).any()

    def test_ones_and_zeros(self):
        # Create arrays of ones and zeros and unwrap
        ones = tools.unwrap_phase_image(np.ones(np.shape(self.array)))
        zeros = tools.unwrap_phase_image(np.zeros(np.shape(self.array)))
        # Check that zero arrays stay the same and that ones array is doubled
        assert (ones == 2 * np.ones(np.shape(self.array))).all()
        assert (zeros == np.zeros(np.shape(self.array))).all()

    def test_input_array_type_assertion(self):
        with pytest.raises(TypeError):
            # No input argument
            tools.unwrap_phase_image(None)
            # Other types
            tools.unwrap_phase_image(list([-5, -4, -3, -2, -1, 0, 1, 2, 3]))
            tools.unwrap_phase_image("abcdef")


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
        print(tools.image_stats(pi_range_calculated))
        np.testing.assert_allclose(tools.image_stats(pi_range_calculated),
                                   self.gold_standard, rtol=1e-7, atol=1e-9)

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
        with pytest.raises(ValueError):
            # Empty array
            tools.convert_to_pi_range(np.array([]))
        with pytest.raises(TypeError):
            # No input argument
            tools.convert_to_pi_range(None)
            # Other errors and types
            tools.convert_to_pi_range("abcdef")


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
        with pytest.raises(ValueError):
            # Wrong type `slices`
            tools.mask_slices(self.shape, np.array([1, 2]))
            # Not all elements of slices are `ints`
            tools.mask_slices(self.shape, [1, 2.2])
            tools.mask_slices(self.shape, 2.2)

    def test_slice_ranges_assertion(self):
        with pytest.raises(ValueError):
            # Slices out of range tests
            tools.mask_slices(self.shape, [0, 3])
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


if __name__ == '__main__':
    pytest.main()
