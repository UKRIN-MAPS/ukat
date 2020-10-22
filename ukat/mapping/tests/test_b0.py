import numpy as np
import pytest
from ukat.mapping.b0 import B0
import ukat.utils.tools as tools

class TestB0:
    # Create arrays for testing
    correct_array = np.arange(200).reshape((10, 10, 2))
    # `correct_array` is wrapped using the algorithm in
    # https://scikit-image.org/docs/dev/auto_examples/filters/plot_phase_unwrap.html
    correct_array = np.angle(np.exp(1j * correct_array))
    one_echo_array = np.arange(100).reshape((10, 10, 1))
    multiple_echoes_array = (np.concatenate((correct_array,
                             np.arange(300).reshape((10, 10, 3))), axis=2))
    correct_echo_list = [4, 7]
    one_echo_list = [4]
    multiple_echo_list = [1, 2, 3, 4, 5]

    # Gold standard: [mean, std, min, max] of B0 when input = `correct_array`
    gold_standard = [386.3850, 0.0, 386.3850, 386.3850]

    def test_b0map_values(self):
        b0_map_calculated = B0(self.correct_array, self.correct_echo_list).b0_map
        np.testing.assert_allclose(tools.image_stats(b0_map_calculated),
                                   self.gold_standard, rtol=1e-7, atol=1e-9)

    def test_inputs(self):
        # Check that it fails when input pixel_array has incorrect shape
        with pytest.raises(ValueError):
            B0(self.one_echo_array, self.correct_echo_list)
        with pytest.raises(ValueError):
            B0(self.multiple_echoes_array, self.correct_echo_list)

        # Check that it fails when input echo_list has incorrect shape
        with pytest.raises(ValueError):
            B0(self.correct_array, self.one_echo_list)
        with pytest.raises(ValueError):
            B0(self.correct_array, self.multiple_echo_list)

        # And when both input pixel_array and echo_list have incorrect shapes
        with pytest.raises(ValueError):
            B0(self.one_echo_array, self.one_echo_list)
        with pytest.raises(ValueError):
            B0(self.multiple_echoes_array, self.one_echo_list)
        with pytest.raises(ValueError):
            B0(self.one_echo_array, self.multiple_echo_list)
        with pytest.raises(ValueError):
            B0(self.multiple_echoes_array, self.multiple_echo_list)

    def test_mask(self):
        # Create a mask where one of the echoes is True and the other is False
        mask = np.ones(self.correct_array.shape[:-1], dtype=bool)
        mask[:5, ...] = False

        all_pixels = B0(self.correct_array, self.correct_echo_list)
        masked_pixels = B0(self.correct_array, self.correct_echo_list, mask=mask)

        assert (all_pixels.phase_difference != masked_pixels.phase_difference).any()
        assert (all_pixels.b0_map != masked_pixels.b0_map).any()
        assert (tools.image_stats(all_pixels.b0_map) != 
                tools.image_stats(masked_pixels.b0_map))

    def test_unwrap_phase(self):
        unwrapped = B0(self.correct_array, self.correct_echo_list)
        wrapped = B0(self.correct_array, self.correct_echo_list, unwrap=False)

        assert (unwrapped.phase_difference != wrapped.phase_difference).any()
        assert (unwrapped.b0_map != wrapped.b0_map).any()
        assert (tools.image_stats(unwrapped.b0_map) != 
                tools.image_stats(wrapped.b0_map))

    def test_pixel_array_type_assertion(self):
        # Empty array
        with pytest.raises(ValueError):
            B0(np.array([]), self.correct_echo_list)
        # No input argument
        with pytest.raises(AttributeError):
            B0(None, self.correct_echo_list)
        # List
        with pytest.raises(AttributeError):
            B0(list([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
                self.correct_echo_list)
        # String
        with pytest.raises(AttributeError):
            B0("abcdef", self.correct_echo_list)

    def test_echo_list_type_assertion(self):
        # Empty list
        with pytest.raises(ValueError):
            B0(self.correct_array, np.array([]))
        # No input argument
        with pytest.raises(TypeError):
            B0(self.correct_array, None)
        # Float
        with pytest.raises(TypeError):
            B0(self.correct_array, 3.2)
        # String
        with pytest.raises(ValueError):
            B0(self.correct_array, "abcdef")
