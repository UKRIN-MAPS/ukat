import os
import shutil
import numpy as np
import numpy.testing as npt
import pytest
from ukat.data import fetch
from ukat.mapping.phase_contrast import PhaseContrast, phase_to_velocity
from ukat.utils import arraystats


class TestPC:
    # Create arrays for testing
    correct_signal = np.arange(2000).reshape((10, 10, 20)) # velocity array
    two_dim_signal = np.arange(100).reshape((10, 10))
    vel_encoding = 100
    affine = np.eye(4)

    # Gold standard: [mean, std, min, max] of velocity_array when input
    # is `correct_signal`
    gold_standard = [999.5, 577.3501970208376, 0, 1999]

    # Expected average and peak velocity and flow values
    mean_vel_cycle_standard = [990.0, 991.0, 992.0, 993.0, 994.0, 995.0, 996.0,
                               997.0, 998.0, 999.0, 1000.0, 1001.0, 1002.0,
                               1003.0, 1004.0, 1005.0, 1006.0, 1007.0, 1008.0,
                               1009.0]
    mean_vel_standard = 999.5
    peak_vel_cycle_standard = [1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987,
                               1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995,
                               1996, 1997, 1998, 1999]
    peak_vel_standard = 1999
    rbf = [59400.0, 59460.0, 59520.0, 59580.0, 59640.0, 59700.0, 59760.0,
           59820.0, 59880.0, 59940.0, 60000.0, 60060.0, 60120.0, 60180.0,
           60240.0, 60300.0, 60360.0, 60420.0, 60480.0, 60540.0]
    mean_rbf = 59970.0

    # Velocity array gold standard stats assuming the `correct_signal`
    # is a phase image.
    artificial_vel_array_standard = [0.0, 181.47064907546604,
                                     -314.1592653589793, 314.1592653589793]

    def test_velocity_array(self):
        vel_array_whole_mask = PhaseContrast(self.correct_signal,
                                             self.affine).velocity_array
        vel_stats = arraystats.ArrayStats(vel_array_whole_mask).calculate()
        npt.assert_allclose([vel_stats["mean"]["3D"], vel_stats["std"]["3D"],
                            vel_stats["min"]["3D"], vel_stats["max"]["3D"]],
                            self.gold_standard, rtol=1e-7, atol=1e-9)

    def test_output_parameters(self):
        mapper = PhaseContrast(self.correct_signal, self.affine)
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
            PhaseContrast(self.two_dim_signal, self.affine)
    
    def test_affine_pixel_spacing(self):
        mapper = PhaseContrast(self.correct_signal, self.affine)
        npt.assert_allclose(mapper.pixel_spacing, [1.0, 1.0],
                            rtol=1e-7, atol=1e-9)

        affine_test_1 = self.affine
        affine_test_1[:, 0] = [1, 2, 2, 0]
        mapper = PhaseContrast(self.correct_signal, affine_test_1)
        npt.assert_allclose(mapper.pixel_spacing, [1.0, 3.0],
                            rtol=1e-7, atol=1e-9)
        
        affine_test_2 = affine_test_1
        affine_test_2[:, 1] = [2, 1, 2, 0]
        mapper = PhaseContrast(self.correct_signal, affine_test_2)
        npt.assert_allclose(mapper.pixel_spacing, [3.0, 3.0],
                            rtol=1e-7, atol=1e-9)

    def test_mask(self):
        # Create a mask where the first 5 rows of each cardiac cycle
        # are False and the rest is True.
        mask = np.ones(self.correct_signal.shape, dtype=bool)
        mask[:5, ...] = False

        all_pixels = PhaseContrast(self.correct_signal, self.affine)
        masked_pixels = PhaseContrast(self.correct_signal,
                                      self.affine, mask=mask)

        assert (all_pixels.velocity_array !=
                masked_pixels.velocity_array).any()
        assert (all_pixels.mean_velocity_cardiac_cycle != 
                masked_pixels.mean_velocity_cardiac_cycle)
        assert all_pixels.mean_velocity != masked_pixels.mean_velocity
        assert all_pixels.RBF != masked_pixels.RBF
        assert all_pixels.mean_RBF != masked_pixels.mean_RBF

    def test_to_nifti(self):
        # Create a PC instance and test different export to NIFTI scenarios
        mapper = PhaseContrast(self.correct_signal, self.affine)

        os.makedirs('test_output', exist_ok=True)

        # Check all is saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='pctest', maps='all')
        output_files = os.listdir('test_output')
        assert len(output_files) == 2
        assert 'pctest_velocity_array.nii.gz' in output_files
        assert 'pctest_mask.nii.gz' in output_files

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
            mapper = PhaseContrast(self.correct_signal, self.affine)
            mapper.to_nifti(output_directory='test_output',
                            base_file_name='pctest', maps='')

        # Delete 'test_output' folder
        shutil.rmtree('test_output')

    def test_phase_to_velocity(self):
        # Assuming that correct_signal is a phase image and not velocity array.
        vel_array_calculated = phase_to_velocity(self.correct_signal,
                                                 self.vel_encoding)
        vel_stats = arraystats.ArrayStats(vel_array_calculated).calculate()
        npt.assert_allclose([vel_stats["mean"]["3D"], vel_stats["std"]["3D"],
                            vel_stats["min"]["3D"], vel_stats["max"]["3D"]],
                            self.artificial_vel_array_standard,
                            rtol=1e-7, atol=1e-9)

    def test_real_data(self):
        # Get left PC test data
        _, velocity, mask, affine, _ = fetch.phase_contrast_left_philips()

        # Gold standard statistics
        gold_standard_left = [0.009531885391710501, 0.5249643990944701,
                              0.0, 61.17216385900974]
        rbf_mean_standard_left = 448.3799164652045

        mapper = PhaseContrast(velocity, affine, mask=mask)
        vel_stats = arraystats.ArrayStats(mapper.velocity_array).calculate()
        npt.assert_allclose([vel_stats["mean"]["3D"], vel_stats["std"]["3D"],
                            vel_stats["min"]["3D"], vel_stats["max"]["3D"]],
                            gold_standard_left, rtol=0.01, atol=0)
        npt.assert_allclose(mapper.mean_RBF, rbf_mean_standard_left, 
                            rtol=0.01, atol=0)

        # Get right PC test data
        _, velocity, mask, affine, _ = fetch.phase_contrast_right_philips()

        # Gold standard statistics
        gold_standard_right = [-0.01206081449306819, 0.7170318603118782,
                               -95.2625151835382, 97.36264065280557]
        rbf_mean_standard_right = -567.3407590309374

        mapper = PhaseContrast(velocity, affine, mask=mask)
        vel_stats = arraystats.ArrayStats(mapper.velocity_array).calculate()
        npt.assert_allclose([vel_stats["mean"]["3D"], vel_stats["std"]["3D"],
                            vel_stats["min"]["3D"], vel_stats["max"]["3D"]],
                            gold_standard_right, rtol=0.01, atol=0)
        npt.assert_allclose(mapper.mean_RBF, rbf_mean_standard_right, 
                            rtol=0.01, atol=0)

# Delete the NIFTI test folder recursively if any of the unit tests failed
if os.path.exists('test_output'):
    shutil.rmtree('test_output')
