import numpy as np
import numpy.testing as npt

from ukat.segment import Segmentation


class TestSegment:
    pixel_array = np.linspace(0, 1, 25).reshape((5, 5))
    affine = np.eye(4)
    mask = pixel_array < 0.5
    segment = Segmentation(pixel_array, affine, mask)

    def test_get_mask(self):
        npt.assert_array_equal(self.segment.get_mask(), self.mask)

    def test_tvk(self):
        assert self.segment.tkv() == 12.0
