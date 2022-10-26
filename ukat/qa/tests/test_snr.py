import numpy as np
import numpy.testing as npt
import os
import shutil
from ukat.data import fetch
from ukat.qa import snr
from ukat.utils import arraystats


class TestIsnr:

    def test_automatic_masking(self):
        # T2W data
        gold_standard_noise_mask = [0.235846, 0.424526, 0.0, 1.0]
        gold_standard_isnr_map = [35.227491, 37.936114, 0.0, 439.747471]
        data, affine = fetch.t2w_volume_philips()
        isnr_obj = snr.Isnr(data, affine)
        assert isnr_obj.noise_mask.shape == data.shape
        assert isnr_obj.isnr_map.shape == data.shape
        noise_mask_stats = arraystats.ArrayStats(
            isnr_obj.noise_mask).calculate()
        isnr_map_stats = arraystats.ArrayStats(isnr_obj.isnr_map).calculate()
        npt.assert_allclose([noise_mask_stats['mean']['3D'],
                             noise_mask_stats['std']['3D'],
                             noise_mask_stats['min']['3D'],
                             noise_mask_stats[ 'max']['3D']],
                            gold_standard_noise_mask, rtol=1e-6, atol=1e-4)
        npt.assert_allclose(isnr_obj.isnr, 45.968827)
        npt.assert_allclose([isnr_map_stats['mean']['3D'],
                             isnr_map_stats['std']['3D'],
                             isnr_map_stats['min']['3D'],
                             isnr_map_stats['max']['3D']],
                            gold_standard_isnr_map, rtol=1e-6, atol=1e-4)

        # T1W data
        gold_standard_noise_mask = [0.251823, 0.43406, 0.0, 1.0]
        gold_standard_isnr_map = [43.397967, 34.232043, 0.0, 217.643822]
        data, affine = fetch.t1w_volume_philips()
        isnr_obj = snr.Isnr(data, affine)
        assert isnr_obj.noise_mask.shape == data.shape
        assert isnr_obj.isnr_map.shape == data.shape
        noise_mask_stats = arraystats.ArrayStats(
            isnr_obj.noise_mask).calculate()
        isnr_map_stats = arraystats.ArrayStats(isnr_obj.isnr_map).calculate()
        npt.assert_allclose([noise_mask_stats['mean']['3D'],
                             noise_mask_stats['std']['3D'],
                             noise_mask_stats['min']['3D'],
                             noise_mask_stats['max']['3D']],
                            gold_standard_noise_mask, rtol=1e-6, atol=1e-4)
        npt.assert_allclose(isnr_obj.isnr, 57.847968)
        npt.assert_allclose([isnr_map_stats['mean']['3D'],
                             isnr_map_stats['std']['3D'],
                             isnr_map_stats['min']['3D'],
                             isnr_map_stats['max']['3D']],
                            gold_standard_isnr_map, rtol=1e-6, atol=1e-4)

        # T2star data
        gold_standard_noise_mask = [0.497915, 0.499996, 0.0, 1.0]
        gold_standard_isnr_map = [9.029904, 14.132268, 0.0, 109.629515]
        data, affine, te = fetch.t2star_philips()
        isnr_obj = snr.Isnr(data, affine)
        assert isnr_obj.noise_mask.shape == data.shape
        assert isnr_obj.isnr_map.shape == data.shape
        noise_mask_stats = arraystats.ArrayStats(
            isnr_obj.noise_mask).calculate()
        isnr_map_stats = arraystats.ArrayStats(isnr_obj.isnr_map).calculate()
        npt.assert_allclose([noise_mask_stats['mean']['4D'],
                             noise_mask_stats['std']['4D'],
                             noise_mask_stats['min']['4D'],
                             noise_mask_stats['max']['4D']],
                            gold_standard_noise_mask, rtol=1e-6, atol=1e-4)
        npt.assert_allclose(isnr_obj.isnr, 17.096477)
        npt.assert_allclose([isnr_map_stats['mean']['4D'],
                             isnr_map_stats['std']['4D'],
                             isnr_map_stats['min']['4D'],
                             isnr_map_stats['max']['4D']],
                            gold_standard_isnr_map, rtol=1e-6, atol=1e-4)

    def test_manual_noise_mask(self):
        np.random.seed(0)
        background = np.random.randn(128, 64, 5) * 25 + 50
        signal = np.random.randn(128, 64, 5) * 150 + 1000
        data = np.concatenate((background, signal), axis=1)
        affine = np.eye(4)

        mask = np.ones(data.shape, dtype=bool)
        mask[:, 64:, :] = False

        isnr_obj = snr.Isnr(data, affine, mask)
        assert isnr_obj.noise_mask.shape == data.shape
        npt.assert_allclose(isnr_obj.isnr, 26.307718)

    def test_different_clusters(self):
        data, affine = fetch.t2w_volume_philips()

        # Three components (default)
        gold_standard_bg = [0.235846, 0.424526, 0.0, 1.0]
        gold_standard_clusters = [0.936891, 0.636081, 0.0, 2.0]
        isnr_obj = snr.Isnr(data, affine)
        assert isnr_obj.noise_mask.shape == data.shape
        isnr_stats_bg = arraystats.ArrayStats(isnr_obj.noise_mask).calculate()
        npt.assert_allclose([isnr_stats_bg['mean']['3D'],
                             isnr_stats_bg['std']['3D'],
                             isnr_stats_bg['min']['3D'],
                             isnr_stats_bg['max']['3D']],
                            gold_standard_bg, rtol=1e-6, atol=1e-4)
        isnr_stats_clusters = arraystats.ArrayStats(
            isnr_obj.clusters).calculate()
        npt.assert_allclose([isnr_stats_clusters['mean']['3D'],
                             isnr_stats_clusters['std']['3D'],
                             isnr_stats_clusters['min']['3D'],
                             isnr_stats_clusters['max']['3D']],
                            gold_standard_clusters, rtol=1e-6, atol=1e-4)
        npt.assert_allclose(isnr_obj.isnr, 45.968827)

        # Four components
        gold_standard_bg = [0.240496, 0.427385, 0.0, 1.0]
        gold_standard_clusters = [1.164344, 0.994117, 0.0, 3.0]
        isnr_obj = snr.Isnr(data, affine, n_clusters=4)
        assert isnr_obj.noise_mask.shape == data.shape
        isnr_stats_bg = arraystats.ArrayStats(isnr_obj.noise_mask).calculate()
        npt.assert_allclose([isnr_stats_bg['mean']['3D'],
                             isnr_stats_bg['std']['3D'],
                             isnr_stats_bg['min']['3D'],
                             isnr_stats_bg['max']['3D']],
                            gold_standard_bg, rtol=1e-6, atol=1e-4)
        isnr_stats_clusters = arraystats.ArrayStats(
            isnr_obj.clusters).calculate()
        npt.assert_allclose([isnr_stats_clusters['mean']['3D'],
                             isnr_stats_clusters['std']['3D'],
                             isnr_stats_clusters['min']['3D'],
                             isnr_stats_clusters['max']['3D']],
                            gold_standard_clusters, rtol=1e-6, atol=1e-4)
        npt.assert_allclose(isnr_obj.isnr, 42.08993)

    def test_to_nifti(self):
        data, affine = fetch.t2w_volume_philips()
        isnr_obj = snr.Isnr(data, affine)
        os.makedirs('test_output', exist_ok=True)
        isnr_obj.to_nifti('test_output', base_file_name='T2w')
        output_files = os.listdir('test_output')
        assert len(output_files) == 1
        assert 'T2w_isnr_map.nii.gz' in output_files

        # Delete 'test_output' folder
        shutil.rmtree('test_output')


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
