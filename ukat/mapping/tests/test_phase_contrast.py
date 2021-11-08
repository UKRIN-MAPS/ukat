import os
import shutil
import numpy as np
import numpy.testing as npt
import pytest
from ukat.data import fetch
from ukat.mapping.phase_contrast import PhaseContrast
from ukat.utils import arraystats


class TestPC:
    # Create arrays for testing
    correct_signal = np.arange(2000).reshape((10, 10, 20))
    two_dim_signal = np.arange(100).reshape((10, 10))
    vel_encoding = 100
    affine = np.eye(4)

    # Gold standard: [mean, std, min, max] of velocity_array when input
    # is `correct_signal`
    gold_standard = [13.051648, 108.320512, -280.281686, 53.051648]

    # Expected average and peak velocity and flow values
    mean_vel_cycle_standard = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    mean_vel_standard = 1
    peak_vel_cycle_standard = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    peak_vel_standard = 1
    rbf = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    mean_rbf = 7

    def test_velocity_array(self):
        vel_array_calculated = PhaseContrast(self.correct_signal,
                               self.vel_encoding, self.affine).velocity_array
        vel_stats = arraystats.ArrayStats(vel_array_calculated).calculate()
        npt.assert_allclose([vel_stats["mean"], vel_stats["std"],
                            vel_stats["min"], vel_stats["max"]],
                            self.gold_standard, rtol=1e-7, atol=1e-9)

    def test_output_parameters(self):
        mapper = PhaseContrast(self.correct_signal, self.vel_encoding,
                               self.affine)
        
        npt.assert_allclose(mapper.mean_velocity_cardiac_cycle,
                            self.mean_vel_cycle_standard,
                            rtol=1e-7, atol=1e-9)
        
        npt.assert_allclose(mapper.mean_velocity, self.mean_vel_standard,
                            rtol=1e-7, atol=1e-9)
                    
        npt.assert_allclose(mapper.peak_velocity_cardiac_cycle,
                            self.peak_vel_cycle_standard,
                            rtol=1e-7, atol=1e-9)
        
        npt.assert_allclose(mapper.peak_velocity, self.peak_vel_standard,
                            rtol=1e-7, atol=1e-9)
        
        npt.assert_allclose(mapper.RBF, self.rbf, rtol=1e-7, atol=1e-9)
        
        npt.assert_allclose(mapper.mean_RBF, self.mean_rbf,
                            rtol=1e-7, atol=1e-9)

    def test_input_errors(self):
        # Check that it fails when input pixel_array has incorrect shape
        with pytest.raises(ValueError):
            PhaseContrast(self.two_dim_signal, self.vel_encoding, self.affine)

        # Check that it fails when velocity_encoding is not a number
        with pytest.raises(TypeError):
            PhaseContrast(self.correct_signal, "velocity string", self.affine)
        
        # Check that it fails at Pixel Spacing calculation when affine
        # is not a 4x4 array
        with pytest.raises(AssertionError):
            PhaseContrast(self.correct_signal, self.vel_encoding,
                          self.affine[:3, :3])
    
    def test_affine_pixel_spacing(self):
        affine_test_1 = self.affine

        affine_test_1[:, 0] = [1, 2, 2, 0]

        affine_test_2 = affine_test_1

        affine_test_2[:, 1] = [2, 1, 2, 0]

        mapper = PhaseContrast(self.correct_signal, self.vel_encoding,
                               self.affine)

        npt.assert_allclose(mapper.pixel_spacing, [1.0, 1.0],
                            rtol=1e-7, atol=1e-9)

        mapper = PhaseContrast(self.correct_signal, self.vel_encoding,
                               affine_test_1)

        npt.assert_allclose(mapper.pixel_spacing, [3.0, 1.0],
                            rtol=1e-7, atol=1e-9)
            
        mapper = PhaseContrast(self.correct_signal, self.vel_encoding,
                               affine_test_2)

        npt.assert_allclose(mapper.pixel_spacing, [3.0, 3.0],
                            rtol=1e-7, atol=1e-9)

    def test_mask(self):
        # Create a mask where the first 5 rows of each cardiac cycle
        # are False and the rest is True.
        mask = np.ones(self.correct_signal.shape, dtype=bool)
        mask[:5, ...] = False

        all_pixels = PhaseContrast(self.correct_signal, self.vel_encoding,
                                   self.affine)
        masked_pixels = PhaseContrast(self.correct_signal, self.vel_encoding,
                                      self.affine, mask=mask)

        assert (all_pixels.velocity_array !=
                masked_pixels.velocity_array).any()
        assert (all_pixels.mean_velocity_cardiac_cycle !=
                masked_pixels.mean_velocity_cardiac_cycle).any()
        assert all_pixels.mean_velocity != masked_pixels.mean_velocity
        assert (all_pixels.RBF != masked_pixels.RBF).any()
        assert all_pixels.mean_RBF != masked_pixels.mean_RBF

    def test_to_nifti(self):
        # Create a PC instance and test different export to NIFTI scenarios
        mapper = PhaseContrast(self.correct_signal, self.vel_encoding,
                               self.affine)

        os.makedirs('test_output', exist_ok=True)

        # Check all is saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='pctest', maps='all')
        output_files = os.listdir('test_output')
        assert len(output_files) == 3
        assert 'pctest_phase_array.nii.gz' in output_files
        assert 'pctest_mask.nii.gz' in output_files
        assert 'pctest_velocity_array.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        # Check that no files are saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='pctest', maps=[])
        output_files = os.listdir('test_output')
        assert len(output_files) == 0

        # Check that only velocity_array saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='pctest', maps=['velocity_array'])
        output_files = os.listdir('test_output')
        assert len(output_files) == 1
        assert 'pctest_velocity_array.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        # Check that it fails when no maps are given
        with pytest.raises(ValueError):
            mapper = PhaseContrast(self.correct_signal, self.vel_encoding,
                                   self.affine)
            mapper.to_nifti(output_directory='test_output',
                            base_file_name='pctest', maps='')

        # Delete 'test_output' folder
        shutil.rmtree('test_output')

    def test_real_data(self):
        # Get test data
        _, phase, mask, affine, velocity_encoding = fetch.phase_contrast_philips_left()

        # Gold standard statistics
        gold_standard_left = [-34.174984, 189.285260, -1739.886907, 786.965213]

        mapper = PhaseContrast(phase, velocity_encoding, affine, mask=mask)
        vel_stats = arraystats.ArrayStats(mapper.velocity_array).calculate()
        npt.assert_allclose([vel_stats["mean"], vel_stats["std"],
                            vel_stats["min"], vel_stats["max"]],
                            gold_standard_left, rtol=0.01, atol=0)


# Delete the NIFTI test folder recursively if any of the unit tests failed
if os.path.exists('test_output'):
    shutil.rmtree('test_output')
