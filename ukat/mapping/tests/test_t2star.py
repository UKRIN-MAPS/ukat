import os
import shutil
import numpy as np
import numpy.testing as npt
import pytest
from ukat.data import fetch
from ukat.mapping.t2star import T2Star, two_param_eq
from ukat.utils import arraystats


class TestT2Star:
    t2star = 50
    m0 = 3000
    t = np.arange(5, 39, 3)

    # The idea signal produced by the equation M8 * exp(-t / T2*) where
    # M0 = 5000 and T2* = 50 ms at 12 echo times between 5 and 38 ms
    correct_signal = np.array([2714.51225411, 2556.4313669, 2407.55639389,
                               2267.35122437, 2135.31096829, 2010.96013811,
                               1893.85093652, 1783.56164391, 1679.6950997,
                               1581.87727213, 1489.75591137, 1402.99928103])
    affine = np.eye(4)

    def test_two_param_eq(self):
        signal = two_param_eq(self.t, self.t2star, self.m0)
        npt.assert_allclose(signal, self.correct_signal, rtol=1e-6, atol=1e-8)

    def test_loglin_fit(self):
        # Make the signal into a 4D array
        signal_array = np.tile(self.correct_signal, (10, 10, 3, 1))

        # Multithread
        mapper = T2Star(signal_array, self.t, self.affine, method='loglin',
                        multithread=True)
        assert mapper.shape == signal_array.shape[:-1]
        npt.assert_almost_equal(mapper.t2star_map.mean(), self.t2star)
        assert np.isnan(mapper.t2star_err.mean())
        npt.assert_almost_equal(mapper.m0_map.mean(), self.m0)
        npt.assert_almost_equal(mapper.r2star_map().mean(), 1 / self.t2star)
        npt.assert_almost_equal(mapper.r2.mean(), 1)

        # Single Threaded
        mapper = T2Star(signal_array, self.t, self.affine, method='loglin',
                        multithread=False)
        assert mapper.shape == signal_array.shape[:-1]
        npt.assert_almost_equal(mapper.t2star_map.mean(), self.t2star)
        assert np.isnan(mapper.t2star_err.mean())
        npt.assert_almost_equal(mapper.m0_map.mean(), self.m0)
        npt.assert_almost_equal(mapper.r2star_map().mean(), 1 / self.t2star)
        npt.assert_almost_equal(mapper.r2.mean(), 1)

        # Auto Threaded
        mapper = T2Star(signal_array, self.t, self.affine, method='loglin',
                        multithread='auto')
        assert mapper.shape == signal_array.shape[:-1]
        npt.assert_almost_equal(mapper.t2star_map.mean(), self.t2star)
        assert np.isnan(mapper.t2star_err.mean())
        npt.assert_almost_equal(mapper.m0_map.mean(), self.m0)
        npt.assert_almost_equal(mapper.r2star_map().mean(), 1 / self.t2star)
        npt.assert_almost_equal(mapper.r2.mean(), 1)

    def test_2p_exp_fit(self):
        # Make the signal into a 4D array
        signal_array = np.tile(self.correct_signal, (10, 10, 3, 1))

        # Multithread
        mapper = T2Star(signal_array, self.t, self.affine, method='2p_exp',
                        multithread=True)
        assert mapper.shape == signal_array.shape[:-1]
        npt.assert_almost_equal(mapper.t2star_map.mean(), self.t2star)
        npt.assert_almost_equal(mapper.t2star_err.mean(), 7.395706644238e-11)
        npt.assert_almost_equal(mapper.m0_map.mean(), self.m0)
        npt.assert_almost_equal(mapper.r2star_map().mean(), 1 / self.t2star)
        npt.assert_almost_equal(mapper.r2.mean(), 1)

        # Single Threaded
        mapper = T2Star(signal_array, self.t, self.affine, method='2p_exp',
                        multithread=False)
        assert mapper.shape == signal_array.shape[:-1]
        npt.assert_almost_equal(mapper.t2star_map.mean(), self.t2star)
        npt.assert_almost_equal(mapper.t2star_err.mean(), 7.395706644238e-11)
        npt.assert_almost_equal(mapper.m0_map.mean(), self.m0)
        npt.assert_almost_equal(mapper.r2star_map().mean(), 1 / self.t2star)
        npt.assert_almost_equal(mapper.r2.mean(), 1)

        # Auto Threaded
        mapper = T2Star(signal_array, self.t, self.affine, method='2p_exp',
                        multithread='auto')
        assert mapper.shape == signal_array.shape[:-1]
        npt.assert_almost_equal(mapper.t2star_map.mean(), self.t2star)
        npt.assert_almost_equal(mapper.t2star_err.mean(), 7.395706644238e-11)
        npt.assert_almost_equal(mapper.m0_map.mean(), self.m0)
        npt.assert_almost_equal(mapper.r2star_map().mean(), 1 / self.t2star)
        npt.assert_almost_equal(mapper.r2.mean(), 1)

        # Fail to fit
        mapper = T2Star(signal_array[..., ::-1], self.t, self.affine,
                        method='2p_exp', multithread=True)
        assert mapper.shape == signal_array.shape[:-1]
        # Voxels that fail to fit are set to zero
        npt.assert_almost_equal(mapper.t2star_map.mean(), 0)
        npt.assert_almost_equal(mapper.t2star_err.mean(), 0)
        npt.assert_almost_equal(mapper.r2.mean(), 0)

    def test_mask(self):
        signal_array = np.tile(self.correct_signal, (10, 10, 3, 1))

        # Bool mask
        mask = np.ones(signal_array.shape[:-1], dtype=bool)
        mask[:5, :, :] = False
        mapper = T2Star(signal_array, self.t, self.affine, mask=mask)
        assert mapper.shape == signal_array.shape[:-1]
        npt.assert_almost_equal(mapper.t2star_map[5:, :, :].mean(),
                                self.t2star)
        npt.assert_almost_equal(mapper.t2star_map[:5, :, :].mean(), 0.0)

        # Int mask
        mask = np.ones(signal_array.shape[:-1])
        mask[:5, :, :] = 0
        mapper = T2Star(signal_array, self.t, self.affine, mask=mask)
        assert mapper.shape == signal_array.shape[:-1]
        npt.assert_almost_equal(mapper.t2star_map[5:, :, :].mean(),
                                self.t2star)
        npt.assert_almost_equal(mapper.t2star_map[:5, :, :].mean(), 0.0)

    def test_to_nifti(self):
        # Create a T1 map instance and test different export to NIFTI scenarios
        signal_array = np.tile(self.correct_signal, (10, 10, 3, 1))
        mapper = T2Star(signal_array, self.t, self.affine, method='2p_exp')

        if os.path.exists('test_output'):
            shutil.rmtree('test_output')
        os.makedirs('test_output', exist_ok=True)

        # Check all is saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='t2startest', maps='all')
        output_files = os.listdir('test_output')
        assert len(output_files) == 7
        assert 't2startest_m0_err.nii.gz' in output_files
        assert 't2startest_m0_map.nii.gz' in output_files
        assert 't2startest_mask.nii.gz' in output_files
        assert 't2startest_r2.nii.gz' in output_files
        assert 't2startest_r2star_map.nii.gz' in output_files
        assert 't2startest_t2star_err.nii.gz' in output_files
        assert 't2startest_t2star_map.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        # Check that no files are saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='t2startest', maps=[])
        output_files = os.listdir('test_output')
        assert len(output_files) == 0

        # Check that only t2star and r2star are saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='t2startest', maps=['mask', 't2star',
                                                           'r2star'])
        output_files = os.listdir('test_output')
        assert len(output_files) == 3
        assert 't2startest_mask.nii.gz' in output_files
        assert 't2startest_r2star_map.nii.gz' in output_files
        assert 't2startest_t2star_map.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        # Check that it fails when no maps are given
        with pytest.raises(ValueError):
            mapper = T2Star(signal_array, self.t, self.affine)
            mapper.to_nifti(output_directory='test_output',
                            base_file_name='t2startest', maps='')

        mapper = T2Star(signal_array, self.t, self.affine, method='loglin')

        # Check that error maps arent saved if method is `loglin`
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='t2startest', maps='all')
        output_files = os.listdir('test_output')
        assert len(output_files) == 5
        assert 't2startest_m0_map.nii.gz' in output_files
        assert 't2startest_mask.nii.gz' in output_files
        assert 't2startest_r2.nii.gz' in output_files
        assert 't2startest_r2star_map.nii.gz' in output_files
        assert 't2startest_t2star_map.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        # Check warning is produced if user explicitly asks for
        # t2star_err
        with pytest.warns(UserWarning):
            mapper.to_nifti(output_directory='test_output',
                            base_file_name='t2startest', maps=['t2star',
                                                               't2star_err'])
            output_files = os.listdir('test_output')
            print(output_files)
            assert len(output_files) == 2
            assert 't2startest_t2star_err.nii.gz' in output_files
            assert 't2startest_t2star_map.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        # Check warning is produced if user explicitly asks for
        # t2star_err
        with pytest.warns(UserWarning):
            mapper.to_nifti(output_directory='test_output',
                            base_file_name='t2startest', maps=['m0', 'm0_err'])
            output_files = os.listdir('test_output')
            assert len(output_files) == 2
            assert 't2startest_m0_err.nii.gz' in output_files
            assert 't2startest_m0_map.nii.gz' in output_files

        # Delete 'test_output' folder
        shutil.rmtree('test_output')

    def test_missmatched_raw_data_and_echo_lengths(self):

        with pytest.raises(AssertionError):
            mapper = T2Star(pixel_array=np.zeros((5, 5, 4)),
                            echo_list=np.linspace(0, 2000, 5),
                            affine=self.affine)

        with pytest.raises(AssertionError):
            mapper = T2Star(pixel_array=np.zeros((5, 5, 5)),
                            echo_list=np.linspace(0, 2000, 4),
                            affine=self.affine)

    def test_methods(self):

        # Not a method string
        with pytest.raises(AssertionError):
            mapper = T2Star(pixel_array=np.zeros((5, 5, 5)),
                            echo_list=np.linspace(0, 2000, 5),
                            affine=self.affine, method='magic')

        # Int method
        with pytest.raises(AssertionError):
            mapper = T2Star(pixel_array=np.zeros((5, 5, 5)),
                            echo_list=np.linspace(0, 2000, 5),
                            affine=self.affine, method=0)

    def test_multithread_options(self):

        # Not valid string
        with pytest.raises(AssertionError):
            mapper = T2Star(pixel_array=np.zeros((5, 5, 5)),
                            echo_list=np.linspace(0, 2000, 5),
                            affine=self.affine, multithread='cloud')

    def test_loglin_warning(self):

        # Generate test data with T2* < 20 ms
        signal = two_param_eq(self.t, 10, self.m0)
        signal_array = np.tile(signal, (10, 10, 3, 1))
        with pytest.warns(UserWarning):
            mapper = T2Star(signal_array, self.t, affine=self.affine,
                            method='loglin')

    def test_real_data(self):

        # Get test data
        image, affine, te = fetch.t2star_philips()
        te *= 1000

        # Crop to reduce runtime
        image = image[30:60, 50:90, 2, :]

        # Gold standard statistics for each method
        gold_standard_loglin = [32.727562, 23.862456, 6.739695, 581.272145]
        gold_standard_2p_exp = [30.724469, 22.156475, 0.0, 529.870475]

        # loglin method
        mapper = T2Star(image, te, self.affine, method='loglin')
        t2star_stats = arraystats.ArrayStats(mapper.t2star_map).calculate()
        npt.assert_allclose([t2star_stats["mean"], t2star_stats["std"],
                                    t2star_stats["min"], t2star_stats["max"]],
                                    gold_standard_loglin, rtol=1e-6, atol=1e-4)

        # 2p_exp method
        mapper = T2Star(image, te, self.affine, method='2p_exp')
        t2star_stats = arraystats.ArrayStats(mapper.t2star_map).calculate()
        npt.assert_allclose([t2star_stats["mean"], t2star_stats["std"],
                                    t2star_stats["min"], t2star_stats["max"]],
                                    gold_standard_2p_exp, rtol=1e-6, atol=1e-4)

    def test_get_fit_signal(self):
        # Loglin fit
        signal_array = np.tile(self.correct_signal, (10, 10, 3, 1))

        mapper = T2Star(signal_array, self.t, self.affine,
                        method='loglin', multithread=False)
        fit_signal = mapper.get_fit_signal()
        npt.assert_array_almost_equal(fit_signal, signal_array)

        # Exponential fit
        signal_array = np.tile(self.correct_signal, (10, 10, 3, 1))

        mapper = T2Star(signal_array, self.t, self.affine,
                        method='2p_exp', multithread=False)
        fit_signal = mapper.get_fit_signal()
        npt.assert_array_almost_equal(fit_signal, signal_array)

# Delete the NIFTI test folder recursively if any of the unit tests failed
if os.path.exists('test_output'):
    shutil.rmtree('test_output')
