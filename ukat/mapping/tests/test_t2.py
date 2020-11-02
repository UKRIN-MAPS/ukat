import numpy as np
import numpy.testing as npt
import pytest
from ukat.data import fetch
from ukat.mapping.t2 import T2, two_param_eq
from ukat.utils import arraystats


class TestT2:
    t2 = 120
    m0 = 3000
    t = np.linspace(12, 120, 10)

    # The idea signal produced by the equation M8 * exp(-t / T2) where
    # M0 = 5000 and T2 = 120 ms at 10 echo times between 12 and 120 ms
    correct_signal = np.array([2714.51225411, 2456.19225923, 2222.45466205,
                               2010.96013811, 1819.59197914, 1646.43490828,
                               1489.75591137, 1347.98689235, 1219.70897922,
                               1103.63832351])

    def test_two_param_eq(self):
        signal = two_param_eq(self.t, self.t2, self.m0)
        npt.assert_allclose(signal, self.correct_signal, rtol=1e-6, atol=1e-8)

    def test_2p_exp_fit(self):
        # Make the signal into a 4D array
        signal_array = np.tile(self.correct_signal, (10, 10, 3, 1))

        # Multithread
        mapper = T2(signal_array, self.t, multithread=True)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t2_map.mean() - self.t2 < 0.1
        assert mapper.m0_map.mean() - self.m0 < 0.1
        assert mapper.r2_map().mean() - 1 / self.t2 < 0.1

        # Single Threaded
        mapper = T2(signal_array, self.t, multithread=False)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t2_map.mean() - self.t2 < 0.1
        assert mapper.m0_map.mean() - self.m0 < 0.1
        assert mapper.r2_map().mean() - 1 / self.t2 < 0.1

        # Auto Threaded
        mapper = T2(signal_array, self.t, multithread='auto')
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t2_map.mean() - self.t2 < 0.1
        assert mapper.m0_map.mean() - self.m0 < 0.1
        assert mapper.r2_map().mean() - 1 / self.t2 < 0.1

        # Fail to fit
        mapper = T2(signal_array[..., ::-1], self.t, multithread=True)
        assert mapper.shape == signal_array.shape[:-1]
        # Voxels that fail to fit are set to zero
        assert mapper.t2_map.mean() == 0.0

    def test_mask(self):
        signal_array = np.tile(self.correct_signal, (10, 10, 3, 1))

        # Bool mask
        mask = np.ones(signal_array.shape[:-1], dtype=bool)
        mask[:5, :, :] = False
        mapper = T2(signal_array, self.t, mask=mask)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t2_map[5:, :, :].mean() - self.t2 < 0.1
        assert mapper.t2_map[:5, :, :].mean() < 0.1

        # Int mask
        mask = np.ones(signal_array.shape[:-1])
        mask[:5, :, :] = 0
        mapper = T2(signal_array, self.t, mask=mask)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t2_map[5:, :, :].mean() - self.t2 < 0.1
        assert mapper.t2_map[:5, :, :].mean() < 0.1

    def test_missmatched_raw_data_and_echo_lengths(self):
        with pytest.raises(AssertionError):
            mapper = T2(pixel_array=np.zeros((5, 5, 4)),
                        echo_list=np.linspace(0, 2000, 5))

        with pytest.raises(AssertionError):
            mapper = T2(pixel_array=np.zeros((5, 5, 5)),
                        echo_list=np.linspace(0, 2000, 4))

    def test_real_data(self):
        # Get test data
        image, affine, te = fetch.t2_philips(1)
        te *= 1000
        # Crop to reduce runtime
        image = image[60:90, 30:70, 2, :]

        # Gold standard statistics
        gold_standard = [105.63945, 39.616205,
                         0.0, 568.160604]

        # 2p_exp method
        mapper = T2(image, te)
        t2_stats = arraystats.ArrayStats(mapper.t2_map).calculate()
        npt.assert_allclose([t2_stats["mean"], t2_stats["std"],
                             t2_stats["min"], t2_stats["max"]],
                            gold_standard, rtol=1e-6, atol=1e-4)
