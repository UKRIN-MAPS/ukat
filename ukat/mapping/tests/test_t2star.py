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

    def test_two_param_eq(self):
        signal = two_param_eq(self.t, self.t2star, self.m0)
        npt.assert_allclose(signal, self.correct_signal, rtol=1e-6, atol=1e-8)

    def test_loglin_fit(self):
        # Make the signal into a 4D array
        signal_array = np.tile(self.correct_signal, (10, 10, 3, 1))

        # Multithread
        mapper = T2Star(signal_array, self.t, method='loglin',
                        multithread=True)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t2star_map.mean() - self.t2star < 0.1
        assert mapper.m0_map.mean() - self.m0 < 0.1
        assert mapper.r2star_map().mean() - 1 / self.t2star < 0.1

        # Single Threaded
        mapper = T2Star(signal_array, self.t, method='loglin',
                        multithread=False)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t2star_map.mean() - self.t2star < 0.1
        assert mapper.m0_map.mean() - self.m0 < 0.1
        assert mapper.r2star_map().mean() - 1 / self.t2star < 0.1

        # Auto Threaded
        mapper = T2Star(signal_array, self.t, method='loglin',
                        multithread='auto')
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t2star_map.mean() - self.t2star < 0.1
        assert mapper.m0_map.mean() - self.m0 < 0.1
        assert mapper.r2star_map().mean() - 1 / self.t2star < 0.1

    def test_2p_exp_fit(self):
        # Make the signal into a 4D array
        signal_array = np.tile(self.correct_signal, (10, 10, 3, 1))

        # Multithread
        mapper = T2Star(signal_array, self.t, method='2p_exp',
                        multithread=True)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t2star_map.mean() - self.t2star < 0.1
        assert mapper.m0_map.mean() - self.m0 < 0.1
        assert mapper.r2star_map().mean() - 1 / self.t2star < 0.1

        # Single Threaded
        mapper = T2Star(signal_array, self.t, method='2p_exp',
                        multithread=False)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t2star_map.mean() - self.t2star < 0.1
        assert mapper.m0_map.mean() - self.m0 < 0.1
        assert mapper.r2star_map().mean() - 1 / self.t2star < 0.1

        # Auto Threaded
        mapper = T2Star(signal_array, self.t, method='2p_exp',
                        multithread='auto')
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t2star_map.mean() - self.t2star < 0.1
        assert mapper.m0_map.mean() - self.m0 < 0.1
        assert mapper.r2star_map().mean() - 1 / self.t2star < 0.1

        # Fail to fit
        mapper = T2Star(signal_array[..., ::-1], self.t,
                        method='2p_exp', multithread=True)
        assert mapper.shape == signal_array.shape[:-1]
        # Voxels that fail to fit are set to zero
        assert mapper.t2star_map.mean() == 0.0

    def test_mask(self):
        signal_array = np.tile(self.correct_signal, (10, 10, 3, 1))

        # Bool mask
        mask = np.ones(signal_array.shape[:-1], dtype=bool)
        mask[:5, :, :] = False
        mapper = T2Star(signal_array, self.t, mask=mask)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t2star_map[5:, :, :].mean() - self.t2star < 0.1
        assert mapper.t2star_map[:5, :, :].mean() < 0.1

        # Int mask
        mask = np.ones(signal_array.shape[:-1])
        mask[:5, :, :] = 0
        mapper = T2Star(signal_array, self.t, mask=mask)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t2star_map[5:, :, :].mean() - self.t2star < 0.1
        assert mapper.t2star_map[:5, :, :].mean() < 0.1

    def test_missmatched_raw_data_and_echo_lengths(self):

        with pytest.raises(AssertionError):
            mapper = T2Star(pixel_array=np.zeros((5, 5, 4)),
                            echo_list=np.linspace(0, 2000, 5))

        with pytest.raises(AssertionError):
            mapper = T2Star(pixel_array=np.zeros((5, 5, 5)),
                            echo_list=np.linspace(0, 2000, 4))

    def test_methods(self):

        # Not a method string
        with pytest.raises(AssertionError):
            mapper = T2Star(pixel_array=np.zeros((5, 5, 5)),
                            echo_list=np.linspace(0, 2000, 5),
                            method='magic')

        # Int method
        with pytest.raises(AssertionError):
            mapper = T2Star(pixel_array=np.zeros((5, 5, 5)),
                            echo_list=np.linspace(0, 2000, 5),
                            method=0)

    def test_multithread_options(self):

        # Not valid string
        with pytest.raises(AssertionError):
            mapper = T2Star(pixel_array=np.zeros((5, 5, 5)),
                            echo_list=np.linspace(0, 2000, 5),
                            multithread='cloud')

    def test_loglin_warning(self):

        # Generate test data with T2* < 20 ms
        signal = two_param_eq(self.t, 10, self.m0)
        signal_array = np.tile(signal, (10, 10, 3, 1))
        with pytest.warns(UserWarning):
            mapper = T2Star(signal_array, self.t, method='loglin')

    def test_real_data(self):

        # Get test data
        image, affine, te = fetch.r2star_philips()
        te *= 1000
        # Crop to reduce runtime
        image = image[30:60, 50:90, 2, :]

        # Gold standard statistics for each method
        gold_standard_loglin = [32.2660346964308, 18.499243841743308,
                                0.0, 239.07407841896983]
        gold_standard_2p_exp = [30.724443852557155, 22.156366883080896,
                                0.0, 529.8640757093401]

        # loglin method
        mapper = T2Star(image, te, method='loglin')
        t2star_stats = arraystats.ArrayStats(mapper.t2star_map).calculate()
        np.testing.assert_allclose([t2star_stats["mean"], t2star_stats["std"],
                                    t2star_stats["min"], t2star_stats["max"]],
                                    gold_standard_loglin, rtol=1e-6, atol=1e-4)

        # 2p_exp method
        mapper = T2Star(image, te, method='2p_exp')
        t2star_stats = arraystats.ArrayStats(mapper.t2star_map).calculate()
        np.testing.assert_allclose([t2star_stats["mean"], t2star_stats["std"],
                                    t2star_stats["min"], t2star_stats["max"]],
                                    gold_standard_2p_exp, rtol=1e-6, atol=1e-4)
