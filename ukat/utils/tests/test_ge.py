import numpy as np
import pytest
from ukat.utils.ge import scale_b1


class TestScaleB1:
    unscalled_b1 = np.ones((10, 10, 5)) * 200

    def test_scale_b1(self):
        scaled_b1 = scale_b1(self.unscalled_b1, 20)
        assert np.allclose(scaled_b1, 100)
