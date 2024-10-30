import os
import shutil
import numpy as np
import numpy.testing as npt
import pytest
from ukat.data import fetch
from ukat.mapping.t2 import T2, two_param_eq, three_param_eq
from ukat.utils import arraystats


class TestT2:
    t2 = 120
    m0 = 3000
    t = np.linspace(12, 120, 10)
    b = 250

    # The ideal signal produced by the equation M0 * exp(-t / T2) where
    # M0 = 3000 and T2 = 120 ms at 10 echo times between 12 and 120 ms
    correct_signal = np.array([2714.51225411, 2456.19225923, 2222.45466205,
                               2010.96013811, 1819.59197914, 1646.43490828,
                               1489.75591137, 1347.98689235, 1219.70897922,
                               1103.63832351])
    affine = np.eye(4)

    def test_two_param_eq(self):
        signal = two_param_eq(self.t, self.t2, self.m0)
        npt.assert_allclose(signal, self.correct_signal, rtol=1e-6, atol=1e-8)

    def test_three_param_eq(self):
        signal = three_param_eq(self.t, self.t2, self.m0, self.b)
        npt.assert_allclose(signal, self.correct_signal + self.b,
                            rtol=1e-6, atol=1e-8)

    def test_2p_exp_fit(self):
        # Make the signal into a 4D array
        signal_array = np.tile(self.correct_signal, (10, 10, 3, 1))

        # Multithread
        mapper = T2(signal_array, self.t, self.affine, multithread=True)
        assert mapper.shape == signal_array.shape[:-1]
        npt.assert_almost_equal(mapper.t2_map.mean(), self.t2)
        npt.assert_almost_equal(mapper.m0_map.mean(), self.m0)
        npt.assert_almost_equal(mapper.r2.mean(), 1)

        # Single Threaded
        mapper = T2(signal_array, self.t, self.affine, multithread=False)
        assert mapper.shape == signal_array.shape[:-1]
        npt.assert_almost_equal(mapper.t2_map.mean(), self.t2)
        npt.assert_almost_equal(mapper.m0_map.mean(), self.m0)
        npt.assert_almost_equal(mapper.r2.mean(), 1)

        # Auto Threaded
        mapper = T2(signal_array, self.t, self.affine, multithread='auto')
        assert mapper.shape == signal_array.shape[:-1]
        npt.assert_almost_equal(mapper.t2_map.mean(), self.t2)
        npt.assert_almost_equal(mapper.m0_map.mean(), self.m0)
        npt.assert_almost_equal(mapper.r2.mean(), 1)

        # Fail to fit
        mapper = T2(signal_array[..., ::-1], self.t, self.affine,
                    multithread=True)
        assert mapper.shape == signal_array.shape[:-1]
        # Voxels that fail to fit are set to zero
        assert mapper.t2_map.mean() == 0.0

    def test_3p_exp_fit(self):
        # Make the signal into a 4D array
        signal_array = np.tile(self.correct_signal, (10, 10, 3, 1)) + self.b

        # Multithread
        mapper = T2(signal_array, self.t, self.affine, multithread=True,
                    method='3p_exp')
        assert mapper.shape == signal_array.shape[:-1]
        npt.assert_almost_equal(mapper.t2_map.mean(), self.t2)
        npt.assert_almost_equal(mapper.m0_map.mean(), self.m0)
        npt.assert_almost_equal(mapper.b_map.mean(), self.b)
        npt.assert_almost_equal(mapper.r2.mean(), 1)

        # Single Threaded
        mapper = T2(signal_array, self.t, self.affine, multithread=False,
                    method='3p_exp')
        assert mapper.shape == signal_array.shape[:-1]
        npt.assert_almost_equal(mapper.t2_map.mean(), self.t2)
        npt.assert_almost_equal(mapper.m0_map.mean(), self.m0)
        npt.assert_almost_equal(mapper.b_map.mean(), self.b)
        npt.assert_almost_equal(mapper.r2.mean(), 1)

    def test_threshold_fit(self):
        # Make the signal into a 4D array
        signal_array = np.tile(self.correct_signal, (10, 10, 3, 1))

        # Multithread
        mapper = T2(signal_array, self.t, self.affine, multithread=True,
                    noise_threshold=1300)
        assert mapper.shape == signal_array.shape[:-1]
        npt.assert_almost_equal(mapper.t2_map.mean(), self.t2)
        npt.assert_almost_equal(mapper.m0_map.mean(), self.m0)
        npt.assert_almost_equal(mapper.r2.mean(), 1)

        # Single Threaded
        mapper = T2(signal_array, self.t, self.affine, multithread=False,
                    noise_threshold=1300)
        assert mapper.shape == signal_array.shape[:-1]
        npt.assert_almost_equal(mapper.t2_map.mean(), self.t2)
        npt.assert_almost_equal(mapper.m0_map.mean(), self.m0)
        npt.assert_almost_equal(mapper.r2.mean(), 1)

    def test_mask(self):
        signal_array = np.tile(self.correct_signal, (10, 10, 3, 1))

        # Bool mask
        mask = np.ones(signal_array.shape[:-1], dtype=bool)
        mask[:5, :, :] = False
        mapper = T2(signal_array, self.t, self.affine, mask=mask)
        assert mapper.shape == signal_array.shape[:-1]
        npt.assert_almost_equal(mapper.t2_map[5:, :, :].mean(), self.t2)
        npt.assert_almost_equal(mapper.t2_map[:5, :, :].mean(), 0)

        # Int mask
        mask = np.ones(signal_array.shape[:-1])
        mask[:5, :, :] = 0
        mapper = T2(signal_array, self.t, self.affine, mask=mask)
        assert mapper.shape == signal_array.shape[:-1]
        npt.assert_almost_equal(mapper.t2_map[5:, :, :].mean(), self.t2)
        npt.assert_almost_equal(mapper.t2_map[:5, :, :].mean(), 0)

    def test_mismatched_raw_data_and_echo_lengths(self):
        with pytest.raises(AssertionError):
            mapper = T2(pixel_array=np.zeros((5, 5, 4)),
                        echo_list=np.linspace(0, 2000, 5),
                        affine=self.affine)

        with pytest.raises(AssertionError):
            mapper = T2(pixel_array=np.zeros((5, 5, 5)),
                        echo_list=np.linspace(0, 2000, 4),
                        affine=self.affine)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            mapper = T2(pixel_array=np.zeros((5, 5, 4)),
                        echo_list=np.linspace(0, 2000, 4),
                        affine=self.affine, method='4p_exp')

    def test_real_data(self):
        # Get test data
        image, affine, te = fetch.t2_philips(1)
        te *= 1000
        # Crop to reduce runtime
        image = image[60:90, 30:70, 2, :]

        # Gold standard statistics
        gold_standard_2p_exp = [105.63945, 39.616205,
                                0.0, 568.160604]
        gold_standard_3p_exp = [98.812108, 42.945342, 0.0, 568.160625]
        gold_standard_thresh = [106.351332, 39.904419,
                                0.0, 568.160832]

        # 2p_exp method
        mapper = T2(image, te, self.affine)
        t2_stats = arraystats.ArrayStats(mapper.t2_map).calculate()
        npt.assert_allclose([t2_stats["mean"], t2_stats["std"],
                             t2_stats["min"], t2_stats["max"]],
                            gold_standard_2p_exp, rtol=1e-6, atol=1e-4)

        # 3p_exp method
        mapper = T2(image, te, self.affine, method='3p_exp')
        t2_stats = arraystats.ArrayStats(mapper.t2_map).calculate()
        npt.assert_allclose([t2_stats["mean"], t2_stats["std"],
                             t2_stats["min"], t2_stats["max"]],
                            gold_standard_3p_exp, rtol=1e-6, atol=1e-4)

        # threshold method
        mapper = T2(image, te, self.affine, noise_threshold=100)
        t2_stats = arraystats.ArrayStats(mapper.t2_map).calculate()
        npt.assert_allclose([t2_stats["mean"], t2_stats["std"],
                             t2_stats["min"], t2_stats["max"]],
                            gold_standard_thresh, rtol=1e-6, atol=1e-4)

    def test_to_nifti(self):
        # Create a T2 map instance and test different export to NIFTI scenarios
        signal_array = np.tile(self.correct_signal, (10, 10, 3, 1))
        mapper = T2(signal_array, self.t, self.affine, method='3p_exp')

        if os.path.exists('test_output'):
            shutil.rmtree('test_output')
        os.makedirs('test_output', exist_ok=True)

        # Check all is saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='t2test', maps='all')
        output_files = os.listdir('test_output')
        assert len(output_files) == 8
        assert 't2test_b_map.nii.gz' in output_files
        assert 't2test_b_err.nii.gz' in output_files
        assert 't2test_m0_err.nii.gz' in output_files
        assert 't2test_m0_map.nii.gz' in output_files
        assert 't2test_mask.nii.gz' in output_files
        assert 't2test_r2_map.nii.gz' in output_files
        assert 't2test_t2_err.nii.gz' in output_files
        assert 't2test_t2_map.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        # Check that no files are saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='t2test', maps=[])
        output_files = os.listdir('test_output')
        assert len(output_files) == 0

        # Check that only mask, t2 and r2 are saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='t2test', maps=['mask', 't2', 'r2'])
        output_files = os.listdir('test_output')
        assert len(output_files) == 3
        assert 't2test_mask.nii.gz' in output_files
        assert 't2test_t2_map.nii.gz' in output_files
        assert 't2test_r2_map.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        # Check that it fails when no maps are given
        with pytest.raises(ValueError):
            mapper = T2(signal_array, self.t, self.affine)
            mapper.to_nifti(output_directory='test_output',
                            base_file_name='t2test', maps='')

        # Delete 'test_output' folder
        shutil.rmtree('test_output')

    def test_get_fit_signal(self):
        # Two parameter fit
        signal_array = np.tile(self.correct_signal, (10, 10, 3, 1))

        mapper = T2(signal_array, self.t, self.affine, multithread=False)
        fit_signal = mapper.get_fit_signal()
        npt.assert_array_almost_equal(fit_signal, signal_array)

        # Three parameter fit
        signal_array = np.tile(self.correct_signal, (10, 10, 3, 1)) + self.b

        mapper = T2(signal_array, self.t, self.affine, multithread=False,
                    method='3p_exp')
        fit_signal = mapper.get_fit_signal()
        npt.assert_array_almost_equal(fit_signal, signal_array)


# Delete the NIFTI test folder recursively if any of the unit tests failed
if os.path.exists('test_output'):
    shutil.rmtree('test_output')
