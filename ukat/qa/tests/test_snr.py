import numpy as np
import numpy.testing as npt
from ukat.data import fetch
from ukat.qa import snr
from ukat.utils import arraystats


class TestIsnr:

    def test_automatic_masking(self):
        # T2W data
        gold_standard = [0.235846, 0.424526, 0.0, 1.0]
        data, affine = fetch.t2w_volume_philips()
        isnr_obj = snr.Isnr(data)
        assert isnr_obj.noise_mask.shape == data.shape
        isnr_stats = arraystats.ArrayStats(isnr_obj.noise_mask).calculate()
        npt.assert_allclose([isnr_stats['mean']['3D'], isnr_stats['std']['3D'],
                             isnr_stats['min']['3D'], isnr_stats['max']['3D']],
                            gold_standard, rtol=1e-6, atol=1e-4)
        npt.assert_allclose(isnr_obj.isnr, 45.968827)

        # T1W data
        gold_standard = [0.251823, 0.43406, 0.0, 1.0]
        data, affine = fetch.t1w_volume_philips()
        isnr_obj = snr.Isnr(data)
        assert isnr_obj.noise_mask.shape == data.shape
        isnr_stats = arraystats.ArrayStats(isnr_obj.noise_mask).calculate()
        npt.assert_allclose([isnr_stats['mean']['3D'], isnr_stats['std']['3D'],
                             isnr_stats['min']['3D'], isnr_stats['max']['3D']],
                            gold_standard, rtol=1e-6, atol=1e-4)
        npt.assert_allclose(isnr_obj.isnr, 57.847968)

        # T2star data
        gold_standard = [0.497915, 0.499996, 0.0, 1.0]
        data, affine, te = fetch.t2star_philips()
        isnr_obj = snr.Isnr(data)
        assert isnr_obj.noise_mask.shape == data.shape
        isnr_stats = arraystats.ArrayStats(isnr_obj.noise_mask).calculate()
        npt.assert_allclose([isnr_stats['mean']['4D'], isnr_stats['std']['4D'],
                             isnr_stats['min']['4D'], isnr_stats['max']['4D']],
                            gold_standard, rtol=1e-6, atol=1e-4)
        npt.assert_allclose(isnr_obj.isnr, 17.096477)

    def test_manual_noise_mask(self):
        np.random.seed(0)
        background = np.random.randn(128, 64, 5) * 25 + 50
        signal = np.random.randn(128, 64, 5) * 150 + 1000
        data = np.concatenate((background, signal), axis=1)

        mask = np.ones(data.shape, dtype=bool)
        mask[:, 64:, :] = False

        isnr_obj = snr.Isnr(data, mask)
        assert isnr_obj.noise_mask.shape == data.shape
        npt.assert_allclose(isnr_obj.isnr, 26.307718)

    def test_different_clusters(self):
        data, affine = fetch.t2w_volume_philips()

        # Three components (default)
        gold_standard_bg = [0.235846, 0.424526, 0.0, 1.0]
        gold_standard_clusters = [0.936891, 0.636081, 0.0, 2.0]
        isnr_obj = snr.Isnr(data)
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

        # Two components
        gold_standard_bg = [0.240496, 0.427385, 0.0, 1.0]
        gold_standard_clusters = [1.164344, 0.994117, 0.0, 3.0]
        isnr_obj = snr.Isnr(data, n_clusters=4)
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
