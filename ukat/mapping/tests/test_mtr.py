import os
import shutil
import numpy as np
import numpy.testing as npt
import pytest
from ukat.data import fetch
from ukat.mapping.mtr import MTR
from ukat.utils import arraystats


class TestMTR:
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

    # Gold standard: [mean, std, min, max] of MTR when input = `correct_array`
    gold_standard = [13.051648, 108.320512, -280.281686, 53.051648]

    def test_mtr_calculation_without_unwrapping(self):
        mtr_map_calculated = MTR(self.correct_array,
                               self.correct_echo_list, self.affine,
                               unwrap=False).mtr_map
        mtrmaps_stats = arraystats.ArrayStats(mtr_map_calculated).calculate()
        npt.assert_allclose([mtrmaps_stats["mean"], mtrmaps_stats["std"],
                            mtrmaps_stats["min"], mtrmaps_stats["max"]],
                            self.gold_standard, rtol=1e-7, atol=1e-9)

    def test_inputs(self):
        # Check that it fails when input pixel_array has incorrect shape
        with pytest.raises(ValueError):
            MTR(self.one_echo_array, self.correct_echo_list, self.affine)
        with pytest.raises(ValueError):
            MTR(self.multiple_echoes_array, self.correct_echo_list, self.affine)

        # Check that it fails when input echo_list has incorrect shape
        with pytest.raises(ValueError):
            MTR(self.correct_array, self.one_echo_list, self.affine)
        with pytest.raises(ValueError):
            MTR(self.correct_array, self.multiple_echo_list, self.affine)

        # And when both input pixel_array and echo_list have incorrect shapes
        with pytest.raises(ValueError):
            MTR(self.one_echo_array, self.one_echo_list, self.affine)
        with pytest.raises(ValueError):
            MTR(self.multiple_echoes_array, self.one_echo_list, self.affine)
        with pytest.raises(ValueError):
            MTR(self.one_echo_array, self.multiple_echo_list, self.affine)
        with pytest.raises(ValueError):
            MTR(self.multiple_echoes_array, self.multiple_echo_list,
               self.affine)

    def test_mask(self):
        # Create a mask where one of the echoes is True and the other is False
        mask = np.ones(self.correct_array.shape[:-1], dtype=bool)
        mask[:5, ...] = False

        all_pixels = MTR(self.correct_array, self.correct_echo_list,
                        self.affine)
        masked_pixels = MTR(self.correct_array, self.correct_echo_list,
                           self.affine, mask=mask)

        assert (all_pixels.phase_difference !=
                masked_pixels.phase_difference).any()
        assert (all_pixels.mtr_map != masked_pixels.mtr_map).any()
        assert (arraystats.ArrayStats(all_pixels.mtr_map).calculate() !=
                arraystats.ArrayStats(masked_pixels.mtr_map).calculate())

    def test_unwrap_phase(self):
        unwrapped = MTR(self.correct_array, self.correct_echo_list, self.affine)
        wrapped = MTR(self.correct_array, self.correct_echo_list, self.affine,
                     unwrap=False)

        assert (unwrapped.phase_difference != wrapped.phase_difference).any()
        assert (unwrapped.mtr_map != wrapped.mtr_map).any()
        assert (arraystats.ArrayStats(unwrapped.mtr_map).calculate() !=
                arraystats.ArrayStats(wrapped.mtr_map).calculate())

    def test_to_nifti(self):
        # Create a MTR map instance and test different export to NIFTI scenarios
        mapper = MTR(self.correct_array, self.correct_echo_list,
                    self.affine, unwrap=False)

        os.makedirs('test_output', exist_ok=True)

        # Check all is saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='mtrtest', maps='all')
        output_files = os.listdir('test_output')
        assert len(output_files) == 5
        assert 'mtrtest_mtr_map.nii.gz' in output_files
        assert 'mtrtest_mask.nii.gz' in output_files
        assert 'mtrtest_phase0.nii.gz' in output_files
        assert 'mtrtest_phase1.nii.gz' in output_files
        assert 'mtrtest_phase_difference.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        # Check that no files are saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='mtrtest', maps=[])
        output_files = os.listdir('test_output')
        assert len(output_files) == 0

        # Check that only mtr, phase0 and phase_difference are saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='mtrtest', maps=['mtr', 'phase0',
                                                       'phase_difference'])
        output_files = os.listdir('test_output')
        assert len(output_files) == 3
        assert 'mtrtest_mtr_map.nii.gz' in output_files
        assert 'mtrtest_phase0.nii.gz' in output_files
        assert 'mtrtest_phase_difference.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        # Check that it fails when no maps are given
        with pytest.raises(ValueError):
            mapper = MTR(self.correct_array, self.correct_echo_list,
                        self.affine, unwrap=False)
            mapper.to_nifti(output_directory='test_output',
                            base_file_name='mtrtest', maps='')

        # Delete 'test_output' folder
        shutil.rmtree('test_output')

    def test_pixel_array_type_assertion(self):
        # Empty array
        with pytest.raises(ValueError):
            mapper = MTR(np.array([]), self.correct_echo_list, self.affine)
        # No input argument
        with pytest.raises(AttributeError):
            mapper = MTR(None, self.correct_echo_list, self.affine)
        # List
        with pytest.raises(AttributeError):
            mapper = MTR(list([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
                        self.correct_echo_list, self.affine)
        # String
        with pytest.raises(AttributeError):
            mapper = MTR("abcdef", self.correct_echo_list, self.affine)

    def test_echo_list_type_assertion(self):
        # Empty list
        with pytest.raises(ValueError):
            mapper = MTR(self.correct_array, np.array([]), self.affine)
        # No input argument
        with pytest.raises(TypeError):
            mapper = MTR(self.correct_array, None, self.affine)
        # Float
        with pytest.raises(TypeError):
            mapper = MTR(self.correct_array, 3.2, self.affine)
        # String
        with pytest.raises(ValueError):
            mapper = MTR(self.correct_array, "abcdef", self.affine)

    def test_real_data(self):
        # Get test data
        magnitude, phase, affine, te = fetch.mtr_philips()
        te *= 1000

        # Process on a central slice only
        images = phase[:, :, 4, :]

        # Gold standard statistics
        gold_standard_mtr = [-34.174984, 189.285260, -1739.886907, 786.965213]

        # MTRMap with unwrapping - Consider that unwrapping method may change
        mapper = MTR(images, te, affine, unwrap=True)
        mtrmap_stats = arraystats.ArrayStats(mapper.mtr_map).calculate()
        npt.assert_allclose([mtrmap_stats["mean"], mtrmap_stats["std"],
                            mtrmap_stats["min"], mtrmap_stats["max"]],
                            gold_standard_mtr, rtol=0.01, atol=0)


# Delete the NIFTI test folder recursively if any of the unit tests failed
if os.path.exists('test_output'):
    shutil.rmtree('test_output')
