import os
import numpy.testing as npt
import shutil

from ukat.data import fetch
from ukat.segment.whole_kidney import Segmentation
from ukat.utils import arraystats


class TestSegmentation:
    image, affine = fetch.t2w_volume_philips()
    segmentation = Segmentation(image, affine)

    def test_get_mask(self):
        expected = [0.022764, 0.14915, 0.0, 1.0]
        mask = self.segmentation.get_mask()
        mask_stats = arraystats.ArrayStats(mask).calculate()
        npt.assert_allclose([mask_stats["mean"]["3D"], mask_stats["std"]["3D"],
                             mask_stats["min"]["3D"], mask_stats["max"]["3D"]],
                            expected, rtol=1e-6, atol=1e-4)

    def test_get_kidneys(self):
        expected = [0.034317, 0.237162, 0.0, 2.0]
        mask = self.segmentation.get_kidneys()
        mask_stats = arraystats.ArrayStats(mask).calculate()
        npt.assert_allclose([mask_stats["mean"]["3D"], mask_stats["std"]["3D"],
                             mask_stats["min"]["3D"], mask_stats["max"]["3D"]],
                            expected, rtol=1e-6, atol=1e-4)

    def test_get_left_kidney(self):
        expected = [0.011211, 0.105285, 0.0, 1.0]
        mask = self.segmentation.get_left_kidney()
        mask_stats = arraystats.ArrayStats(mask).calculate()
        npt.assert_allclose([mask_stats["mean"]["3D"], mask_stats["std"]["3D"],
                             mask_stats["min"]["3D"], mask_stats["max"]["3D"]],
                            expected, rtol=1e-6, atol=1e-4)

    def test_get_right_kidney(self):
        expected = [0.011553, 0.106863, 0.0, 1.0]
        mask = self.segmentation.get_right_kidney()
        mask_stats = arraystats.ArrayStats(mask).calculate()
        npt.assert_allclose([mask_stats["mean"]["3D"], mask_stats["std"]["3D"],
                             mask_stats["min"]["3D"], mask_stats["max"]["3D"]],
                            expected, rtol=1e-6, atol=1e-4)

    def test_get_volumes(self):
        expected = {'total': 240.00054654884337,
                    'left': 118.19352480602264,
                    'right': 121.80702174282074}
        volumes = self.segmentation.get_volumes()
        assert volumes == expected

    def test_get_tkv(self):
        expected = 240.00054654884337
        assert self.segmentation.get_tkv() == expected

    def test_get_lkv(self):
        expected = 118.19352480602264
        assert self.segmentation.get_lkv() == expected

    def test_get_rkv(self):
        expected = 121.80702174282074
        assert self.segmentation.get_rkv() == expected
