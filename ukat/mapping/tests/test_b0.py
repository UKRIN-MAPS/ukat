import os
import shutil
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
    affine = np.eye(4)
    correct_echo_list = [4, 7]
    one_echo_list = [4]
    multiple_echo_list = [1, 2, 3, 4, 5]

    # Gold standard: [mean, std, min, max] of B0 when input = `correct_array`
    gold_standard = [13.051648, 108.320512, -280.281686, 53.051648]

    def test_b0_calculation_without_unwrapping(self):
        b0_map_calculated = B0(self.correct_array,
                               self.correct_echo_list, self.affine,
                               unwrap=False).b0_map
        b0maps_stats = arraystats.ArrayStats(b0_map_calculated).calculate()
        npt.assert_allclose([b0maps_stats["mean"], b0maps_stats["std"],
                            b0maps_stats["min"], b0maps_stats["max"]],
                            self.gold_standard, rtol=1e-7, atol=1e-9)

    def test_inputs(self):
        # Check that it fails when input pixel_array has incorrect shape
        with pytest.raises(ValueError):
            B0(self.one_echo_array, self.correct_echo_list, self.affine)
        with pytest.raises(ValueError):
            B0(self.multiple_echoes_array, self.correct_echo_list, self.affine)

        # Check that it fails when input echo_list has incorrect shape
        with pytest.raises(ValueError):
            B0(self.correct_array, self.one_echo_list, self.affine)
        with pytest.raises(ValueError):
            B0(self.correct_array, self.multiple_echo_list, self.affine)

        # And when both input pixel_array and echo_list have incorrect shapes
        with pytest.raises(ValueError):
            B0(self.one_echo_array, self.one_echo_list, self.affine)
        with pytest.raises(ValueError):
            B0(self.multiple_echoes_array, self.one_echo_list, self.affine)
        with pytest.raises(ValueError):
            B0(self.one_echo_array, self.multiple_echo_list, self.affine)
        with pytest.raises(ValueError):
            B0(self.multiple_echoes_array, self.multiple_echo_list,
               self.affine)

    def test_mask(self):
        # Create a mask where one of the echoes is True and the other is False
        mask = np.ones(self.correct_array.shape[:-1], dtype=bool)
        mask[:5, ...] = False

        all_pixels = B0(self.correct_array, self.correct_echo_list,
                        self.affine)
        masked_pixels = B0(self.correct_array, self.correct_echo_list,
                           self.affine, mask=mask)

        assert (all_pixels.phase_difference !=
                masked_pixels.phase_difference).any()
        assert (all_pixels.b0_map != masked_pixels.b0_map).any()
        assert (arraystats.ArrayStats(all_pixels.b0_map).calculate() !=
                arraystats.ArrayStats(masked_pixels.b0_map).calculate())

    def test_unwrap_phase(self):
        unwrapped = B0(self.correct_array, self.correct_echo_list, self.affine)
        wrapped = B0(self.correct_array, self.correct_echo_list, self.affine,
                     unwrap=False)

        assert (unwrapped.phase_difference != wrapped.phase_difference).any()
        assert (unwrapped.b0_map != wrapped.b0_map).any()
        assert (arraystats.ArrayStats(unwrapped.b0_map).calculate() !=
                arraystats.ArrayStats(wrapped.b0_map).calculate())

    def test_to_nifti(self):
        # Create a B0 map instance and test different export to NIFTI scenarios
        mapper = B0(self.correct_array, self.correct_echo_list,
                    self.affine, unwrap=False)

        os.makedirs('test_output', exist_ok=True)

        # Check all is saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='b0test', maps='all')
        output_files = os.listdir('test_output')
        assert len(output_files) == 5
        assert 'b0test_b0_map.nii.gz' in output_files
        assert 'b0test_mask.nii.gz' in output_files
        assert 'b0test_phase0.nii.gz' in output_files
        assert 'b0test_phase1.nii.gz' in output_files
        assert 'b0test_phase_difference.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        # Check that no files are saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='b0test', maps=[])
        output_files = os.listdir('test_output')
        assert len(output_files) == 0

        # Check that only b0, phase0 and phase_difference are saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='b0test', maps=['b0', 'phase0',
                                                       'phase_difference'])
        output_files = os.listdir('test_output')
        assert len(output_files) == 3
        assert 'b0test_b0_map.nii.gz' in output_files
        assert 'b0test_phase0.nii.gz' in output_files
        assert 'b0test_phase_difference.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        # Check that it fails when no maps are given
        with pytest.raises(ValueError):
            mapper = B0(self.correct_array, self.correct_echo_list,
                        self.affine, unwrap=False)
            mapper.to_nifti(output_directory='test_output',
                            base_file_name='b0test', maps='')

        # Delete 'test_output' folder
        shutil.rmtree('test_output')

    def test_pixel_array_type_assertion(self):
        # Empty array
        with pytest.raises(ValueError):
            mapper = B0(np.array([]), self.correct_echo_list, self.affine)
        # No input argument
        with pytest.raises(AttributeError):
            mapper = B0(None, self.correct_echo_list, self.affine)
        # List
        with pytest.raises(AttributeError):
            mapper = B0(list([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
                        self.correct_echo_list, self.affine)
        # String
        with pytest.raises(AttributeError):
            mapper = B0("abcdef", self.correct_echo_list, self.affine)

    def test_echo_list_type_assertion(self):
        # Empty list
        with pytest.raises(ValueError):
            mapper = B0(self.correct_array, np.array([]), self.affine)
        # No input argument
        with pytest.raises(TypeError):
            mapper = B0(self.correct_array, None, self.affine)
        # Float
        with pytest.raises(TypeError):
            mapper = B0(self.correct_array, 3.2, self.affine)
        # String
        with pytest.raises(ValueError):
            mapper = B0(self.correct_array, "abcdef", self.affine)

    def test_real_data(self):
        # Get test data
        magnitude, phase, affine, te = fetch.b0_philips()
        te *= 1000

        # Process on a central slice only
        images = phase[:, :, 4, :]

        # Gold standard statistics
        gold_standard_b0 = [-34.174984, 189.285260, -1739.886907, 786.965213]

        # B0Map with unwrapping - Consider that unwrapping method may change
        mapper = B0(images, te, affine, unwrap=True)
        b0map_stats = arraystats.ArrayStats(mapper.b0_map).calculate()
        npt.assert_allclose([b0map_stats["mean"], b0map_stats["std"],
                            b0map_stats["min"], b0map_stats["max"]],
                            gold_standard_b0, rtol=0.01, atol=0)

    def test_b0_offset_correction(self):
        # Get test data that does not require b0_offset correction
        magnitude, phase, affine, te = fetch.b0_philips()
        te *= 1000
        # Process on a central slice only
        images = phase[:, :, 4, :]
        # B0Map with unwrapping
        mapper = B0(images, te, affine, unwrap=True)
        b0_map_without_offset_correction = (mapper.phase_difference /
                                            (2 * np.pi * mapper.delta_te))
        # This assertion proves that no offset correction was performed
        assert (mapper.b0_map == b0_map_without_offset_correction).all()

        # Get test data that requires b0_offset correction
        magnitude, phase, affine, te = fetch.b0_siemens(1)
        te *= 1000
        # Process on a central slice only
        images = phase[:, :, 4, :]
        # B0Map with unwrapping
        mapper = B0(images, te, affine, unwrap=True)
        b0_map_without_offset_correction = (mapper.phase_difference /
                                            (2 * np.pi * mapper.delta_te))
        # This assertion proves that there was offset correction performed
        assert (mapper.b0_map != b0_map_without_offset_correction).any()

# Delete the NIFTI test folder recursively if any of the unit tests failed
if os.path.exists('test_output'):
    shutil.rmtree('test_output')
