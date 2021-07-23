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
    array_one = np.arange(0, 300, 3).reshape((10, 10, 1))
    array_one[-1] = 0
    array_two = np.arange(0, 200, 2).reshape((10, 10, 1))
    array_two[-1] = 0
    correct_array = np.concatenate((array_one, array_two), axis=2)
    swapped_mts_array = np.concatenate((array_two, array_one), axis=2)
    three_mts_array = np.concatenate((array_one, array_two, array_two), axis=2)
    affine = np.eye(4)

    # Gold standard: [mean, std, min, max] of MTR when input = `correct_array`
    # MTR for comparison is = `(array_two - array_one) / array_one`
    gold_standard_mtr = [0.2966666666, 0.1042965856, 0.0, 0.3333333333]

    def test_mtr_calculation(self):
        mapper = MTR(self.correct_array, self.affine)
        mtr_map_calculated = mapper.mtr_map
        mtrmaps_stats = arraystats.ArrayStats(mtr_map_calculated).calculate()
        npt.assert_allclose([mtrmaps_stats["mean"], mtrmaps_stats["std"],
                            mtrmaps_stats["min"], mtrmaps_stats["max"]],
                            self.gold_standard_mtr, rtol=1e-7, atol=1e-9)
        # The purpose of the next assert is to test the infinite situation
        assert (mtr_map_calculated[-1] ==
                np.zeros(np.shape(mtr_map_calculated[-1]))).all()
        assert (mapper.mt_off == np.squeeze(self.array_one)).all()
        assert (mapper.mt_on == np.squeeze(self.array_two)).all()

    def test_inputs(self):
        # Check that it fails when input pixel_array has incorrect shape
        with pytest.raises(AssertionError):
            MTR(self.array_one, self.affine)
        with pytest.raises(AssertionError):
            MTR(self.three_mts_array, self.affine)
        # Check if there's a warning if sum(mt_off) > sum(mt_on)
        with pytest.warns(UserWarning):
            MTR(self.swapped_mts_array, self.affine)

    def test_mask(self):
        # Create a mask
        mask = np.ones(self.correct_array.shape[:-1], dtype=bool)
        mask[:5] = False

        all_pixels = MTR(self.correct_array, self.affine)
        masked_pixels = MTR(self.correct_array, self.affine, mask=mask)

        assert (all_pixels.mtr_map != masked_pixels.mtr_map).any()
        assert (all_pixels.mt_on != masked_pixels.mt_on).any()
        assert (all_pixels.mt_off != masked_pixels.mt_off).any()
        assert (arraystats.ArrayStats(all_pixels.mtr_map).calculate() !=
                arraystats.ArrayStats(masked_pixels.mtr_map).calculate())

    def test_to_nifti(self):
        # Create a MTR map instance and test different export to NIFTI options.
        mapper = MTR(self.correct_array, self.affine)

        os.makedirs('test_output', exist_ok=True)

        # Check all is saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='mtrtest', maps='all')
        output_files = os.listdir('test_output')
        assert len(output_files) == 4
        assert 'mtrtest_mtr_map.nii.gz' in output_files
        assert 'mtrtest_mt_on.nii.gz' in output_files
        assert 'mtrtest_mt_off.nii.gz' in output_files
        assert 'mtrtest_mask.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        # Check that no files are saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='mtrtest', maps=[])
        output_files = os.listdir('test_output')
        assert len(output_files) == 0

        # Check that only `mtr` and `mt_on` are saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='mtrtest', maps=['mtr', 'mt_on'])
        output_files = os.listdir('test_output')
        assert len(output_files) == 2
        assert 'mtrtest_mtr_map.nii.gz' in output_files
        assert 'mtrtest_mt_on.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        # Check that it fails when no maps are given
        with pytest.raises(ValueError):
            mapper = MTR(self.correct_array, self.affine)
            mapper.to_nifti(output_directory='test_output',
                            base_file_name='mtrtest', maps='')

        # Delete 'test_output' folder
        shutil.rmtree('test_output')

    def test_real_data(self):
        # Get test data
        images, affine = fetch.mtr_philips()

        # Gold standard statistics
        gold_standard_mtr_real = [0.1845690591, 0.6237606679, -73.0, 1.0]
        # The minimum should be 0, but the MT_ON and MT_OFF of real data
        # isn't perfectly aligned, which will result in outliers.

        # MTR Map
        mapper = MTR(images, affine)

        # Stats comparison
        mtrmap_stats = arraystats.ArrayStats(mapper.mtr_map).calculate()
        npt.assert_allclose([mtrmap_stats["mean"], mtrmap_stats["std"],
                            mtrmap_stats["min"], mtrmap_stats["max"]],
                            gold_standard_mtr_real, rtol=0.01, atol=0)


# Delete the NIFTI test folder recursively if any of the unit tests failed
if os.path.exists('test_output'):
    shutil.rmtree('test_output')
