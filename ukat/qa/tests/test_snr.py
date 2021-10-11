import numpy as np
import numpy.testing as npt
from ukat.data import fetch
from ukat.qa import snr
from ukat.utils import arraystats


class TestIsnr:

    def test_automatic_masking(self):
        # T2W data
        gold_standard = [0.256466, 0.436682, 0.0, 1.0]
        data, affine = fetch.t2w_volume_philips()
        isnr_obj = snr.Isnr(data)
        assert isnr_obj.noise_mask.shape == data.shape
        isnr_stats = arraystats.ArrayStats(isnr_obj.noise_mask).calculate()
        npt.assert_allclose([isnr_stats['mean']['3D'], isnr_stats['std']['3D'],
                             isnr_stats['min']['3D'], isnr_stats['max']['3D']],
                            gold_standard, rtol=1e-6, atol=1e-4)
        npt.assert_allclose(isnr_obj.isnr, 31.331225)

        # T1W data
        gold_standard = [0.256569, 0.43674, 0.0, 1.0]
        data, affine = fetch.t1w_volume_philips()
        isnr_obj = snr.Isnr(data)
        assert isnr_obj.noise_mask.shape == data.shape
        isnr_stats = arraystats.ArrayStats(isnr_obj.noise_mask).calculate()
        npt.assert_allclose([isnr_stats['mean']['3D'], isnr_stats['std']['3D'],
                             isnr_stats['min']['3D'], isnr_stats['max']['3D']],
                            gold_standard, rtol=1e-6, atol=1e-4)
        npt.assert_allclose(isnr_obj.isnr, 50.079522)

        # T2star data
        gold_standard = [0.645286, 0.478427, 0.0, 1.0]
        data, affine, te = fetch.t2star_philips()
        isnr_obj = snr.Isnr(data)
        assert isnr_obj.noise_mask.shape == data.shape
        isnr_stats = arraystats.ArrayStats(isnr_obj.noise_mask).calculate()
        npt.assert_allclose([isnr_stats['mean']['4D'], isnr_stats['std']['4D'],
                             isnr_stats['min']['4D'], isnr_stats['max']['4D']],
                            gold_standard, rtol=1e-6, atol=1e-4)
        npt.assert_allclose(isnr_obj.isnr, 11.159309)

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
