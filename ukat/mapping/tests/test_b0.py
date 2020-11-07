import numpy as np
import numpy.testing as npt
import pytest
from ukat.data import fetch
from ukat.mapping.b0 import B0
from ukat.utils import arraystats


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
    gold_standard = [13.051648, 108.320512, -280.281686, 53.051648]

    def test_b0_calculation_without_unwrapping(self):
        b0_map_calculated = B0(self.correct_array,
                               self.correct_echo_list, unwrap=False).b0_map
        b0maps_stats = arraystats.ArrayStats(b0_map_calculated).calculate()
        npt.assert_allclose([b0maps_stats["mean"], b0maps_stats["std"],
                            b0maps_stats["min"], b0maps_stats["max"]],
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
        masked_pixels = B0(self.correct_array, self.correct_echo_list,
                           mask=mask)

        assert (all_pixels.phase_difference !=
                masked_pixels.phase_difference).any()
        assert (all_pixels.b0_map != masked_pixels.b0_map).any()
        assert (arraystats.ArrayStats(all_pixels.b0_map).calculate() !=
                arraystats.ArrayStats(masked_pixels.b0_map).calculate())

    def test_unwrap_phase(self):
        unwrapped = B0(self.correct_array, self.correct_echo_list)
        wrapped = B0(self.correct_array, self.correct_echo_list, unwrap=False)

        assert (unwrapped.phase_difference != wrapped.phase_difference).any()
        assert (unwrapped.b0_map != wrapped.b0_map).any()
        assert (arraystats.ArrayStats(unwrapped.b0_map).calculate() !=
                arraystats.ArrayStats(wrapped.b0_map).calculate())

    def test_pixel_array_type_assertion(self):
        # Empty array
        with pytest.raises(ValueError):
            mapper = B0(np.array([]), self.correct_echo_list)
        # No input argument
        with pytest.raises(AttributeError):
            mapper = B0(None, self.correct_echo_list)
        # List
        with pytest.raises(AttributeError):
            mapper = B0(list([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
                self.correct_echo_list)
        # String
        with pytest.raises(AttributeError):
            mapper = B0("abcdef", self.correct_echo_list)

    def test_echo_list_type_assertion(self):
        # Empty list
        with pytest.raises(ValueError):
            mapper = B0(self.correct_array, np.array([]))
        # No input argument
        with pytest.raises(TypeError):
            mapper = B0(self.correct_array, None)
        # Float
        with pytest.raises(TypeError):
            mapper = B0(self.correct_array, 3.2)
        # String
        with pytest.raises(ValueError):
            mapper = B0(self.correct_array, "abcdef")

    def test_real_data(self):
        # Get test data
        magnitude, phase, affine, te = fetch.b0_philips(2)
        te *= 1000
        # Crop to reduce runtime
        images = phase[30:60, 50:90, 4, :]

        # Gold standard statistics for each method
        gold_standard_b0map = [5.715511, 105.789000, -433.165542, 430.615635]

        # B0Map without unwrapping - unwrapping method may change
        mapper = B0(images, te, unwrap=False)
        b0map_stats = arraystats.ArrayStats(mapper.b0_map).calculate()
        npt.assert_allclose([b0map_stats["mean"], b0map_stats["std"],
                            b0map_stats["min"], b0map_stats["max"]],
                            gold_standard_b0map,  rtol=1e-7, atol=1e-9)
