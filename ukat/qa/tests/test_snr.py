import os
import shutil
import numpy as np
import numpy.testing as npt
from ukat.data import fetch
from ukat.qa import snr
from ukat.utils import arraystats


class TestTsnr:
    np.random.seed(0)
    signal_high = np.random.randn(128, 64, 5, 40) * 50 + 9000
    signal_low = np.random.randn(128, 64, 5, 40) * 150 + 9000
    data = np.concatenate((signal_high, signal_low), axis=1)
    affine = np.eye(4)

    def test_tsnr_synthetic_data(self):
        gold_standard = [127.408817, 65.935981, 42.293117, 370.549169]

        tsnr_obj = snr.Tsnr(self.data, self.affine)
        assert tsnr_obj.shape == self.data.shape[:-1]
        assert tsnr_obj.n_d == self.data.shape[-1]
        tsnr_stats = arraystats.ArrayStats(tsnr_obj.tsnr_map).calculate()
        npt.assert_allclose([tsnr_stats['mean']['3D'], tsnr_stats['std']['3D'],
                             tsnr_stats['min']['3D'], tsnr_stats['max']['3D']],
                            gold_standard, rtol=1e-6, atol=1e-4)

    def test_mask(self):
        gold_standard = [95.566893, 96.903421, 0.0, 370.549169]

        # Bool mask
        mask = np.ones(self.data.shape[:-1], dtype=bool)
        mask[:, 64:, :] = False
        tsnr_obj = snr.Tsnr(self.data, self.affine, mask=mask)
        assert tsnr_obj.shape == self.data.shape[:-1]
        assert tsnr_obj.n_d == self.data.shape[-1]
        tsnr_stats = arraystats.ArrayStats(tsnr_obj.tsnr_map).calculate()
        npt.assert_allclose([tsnr_stats['mean']['3D'], tsnr_stats['std']['3D'],
                             tsnr_stats['min']['3D'], tsnr_stats['max']['3D']],
                            gold_standard, rtol=1e-6, atol=1e-4)

        # Int mask
        mask = np.ones(self.data.shape[:-1])
        mask[:, 64:, :] = 0
        tsnr_obj = snr.Tsnr(self.data, self.affine, mask=mask)
        assert tsnr_obj.shape == self.data.shape[:-1]
        assert tsnr_obj.n_d == self.data.shape[-1]
        tsnr_stats = arraystats.ArrayStats(tsnr_obj.tsnr_map).calculate()
        npt.assert_allclose([tsnr_stats['mean']['3D'], tsnr_stats['std']['3D'],
                             tsnr_stats['min']['3D'], tsnr_stats['max']['3D']],
                            gold_standard, rtol=1e-6, atol=1e-4)

    def test_real_data(self):
        # High tSNR sample data
        gold_standard = [9.436505, 8.775681, -1.163641, 116.984336]
        data, affine = fetch.tsnr_high_philips()
        tsnr_obj = snr.Tsnr(data, affine)
        assert tsnr_obj.shape == data.shape[:-1]
        assert tsnr_obj.n_d == data.shape[-1]
        tsnr_stats = arraystats.ArrayStats(tsnr_obj.tsnr_map).calculate()
        npt.assert_allclose([tsnr_stats['mean']['3D'], tsnr_stats['std']['3D'],
                             tsnr_stats['min']['3D'], tsnr_stats['max']['3D']],
                            gold_standard, rtol=1e-6, atol=1e-4)

        # Low tSNR sample data
        gold_standard = [1.805342, 1.435092, -3.471621, 26.160435]
        data, affine = fetch.tsnr_low_philips()
        tsnr_obj = snr.Tsnr(data, affine)
        assert tsnr_obj.shape == data.shape[:-1]
        assert tsnr_obj.n_d == data.shape[-1]
        tsnr_stats = arraystats.ArrayStats(tsnr_obj.tsnr_map).calculate()
        npt.assert_allclose([tsnr_stats['mean']['3D'], tsnr_stats['std']['3D'],
                             tsnr_stats['min']['3D'], tsnr_stats['max']['3D']],
                            gold_standard, rtol=1e-6, atol=1e-4)

    def test_to_nifti(self):
        tsnr_obj = snr.Tsnr(self.data, self.affine)
        os.makedirs('test_output', exist_ok=True)
        tsnr_obj.to_nifti('test_output', base_file_name='synthetic_data')
        output_files = os.listdir('test_output')
        assert len(output_files) == 1
        assert 'synthetic_data_tsnr_map.nii.gz' in output_files

        # Delete 'test_output' folder
        shutil.rmtree('test_output')


# Delete the NIFTI test folder recursively if any of the unit tests failed
if os.path.exists('test_output'):
    shutil.rmtree('test_output')