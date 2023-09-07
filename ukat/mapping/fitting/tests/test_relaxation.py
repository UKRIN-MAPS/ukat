import numpy as np
import numpy.testing as npt

from ukat.mapping.fitting import Model, fit_image, fit_signal


class TestModel:
    pixel_array = np.zeros((10, 10, 3, 8))
    x = np.linspace(0, 1000, 8)
    mask = np.ones((10, 10, 3), dtype=bool)
    mask[:5] = False

    @staticmethod
    def two_param_eq(x, a, b):
        return a * x + b

    @staticmethod
    def three_param_eq(x, a, b, c):
        return a * x + (b * c)

    def test_init(self):
        model = Model(self.pixel_array, self.x, self.two_param_eq, self.mask,
                      multithread=True)
        assert model.map_shape == (10, 10, 3)
        assert model.n_x == 8

    def test_n_params(self):
        model = Model(self.pixel_array, self.x, self.two_param_eq, self.mask,
                      multithread=True)
        assert model.n_params == 2

        model = Model(self.pixel_array, self.x, self.three_param_eq, self.mask,
                      multithread=True)
        assert model.n_params == 3

    def test_generate_lists(self):
        model = Model(self.pixel_array, self.x, self.two_param_eq, self.mask,
                      multithread=True)
        model.initial_guess = [1, 1]
        model.generate_lists()
        assert type(model.signal_list) == list
        assert type(model.x_list) == list
        assert type(model.p0_list) == list
        assert type(model.mask_list) == list

        assert len(model.signal_list) == 300
        assert len(model.x_list) == 300
        assert len(model.p0_list) == 300
        assert len(model.mask_list) == 300

        assert len(model.signal_list[0]) == 8
        assert len(model.x_list[0]) == 8
        assert len(model.p0_list[0]) == 2

        model = Model(self.pixel_array, self.x, self.two_param_eq,
                      multithread=True)
        model.initial_guess = [1, 1]
        model.generate_lists()
        assert type(model.mask_list) == list
        assert len(model.mask_list) == 300
        assert model.mask_list[0] is True


class TestFitSignal:
    x = np.arange(1, 9)

    @staticmethod
    def linear_eq(x, m, c):
        return m * x + c

    def test_fit_signal(self):
        sig = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        pixel_array = np.tile(sig, (10, 10, 3, 1))
        model = Model(pixel_array, self.x, self.linear_eq,
                      multithread=True)
        model.initial_guess = [0.9, 0.9]
        model.bounds = ([0, 0], [2, 2])
        model.generate_lists()
        popt, error, r2 = fit_signal(sig, self.x, model.initial_guess, True,
                                     model)
        npt.assert_allclose(popt, [1, 0], rtol=1e-5, atol=1e4)
        npt.assert_allclose(error, [0, 0], rtol=1e-5, atol=1e4)
        npt.assert_almost_equal(r2, 1)

    def test_mask(self):
        sig = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        pixel_array = np.tile(sig, (10, 10, 3, 1))
        model = Model(pixel_array, self.x, self.linear_eq,
                      multithread=True)
        model.initial_guess = [0.9, 0.9]
        model.bounds = ([0, 0], [2, 2])
        model.generate_lists()
        popt, error, r2 = fit_signal(sig, self.x, model.initial_guess, False,
                                     model)
        npt.assert_allclose(popt, [0, 0])
        npt.assert_allclose(error, [0, 0])
        npt.assert_almost_equal(r2, -1E6)


class TestFitImage:
    x = np.arange(1, 9)
    sig = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    pixel_array = np.tile(sig, (10, 10, 3, 1))

    @staticmethod
    def linear_eq(x, m, c):
        return m * x + c

    def test_single_threaded(self):
        model = Model(self.pixel_array, self.x, self.linear_eq,
                      multithread=False)
        model.initial_guess = [0.9, 0.9]
        model.bounds = ([0, 0], [2, 2])
        model.generate_lists()
        popt, error, r2 = fit_image(model)

        assert len(popt) == 2
        assert len(error) == 2

        assert popt[0].shape == (10, 10, 3)
        assert error[0].shape == (10, 10, 3)
        assert r2.shape == (10, 10, 3)

        npt.assert_almost_equal(popt[0].mean(), 1, decimal=5)
        npt.assert_almost_equal(error[0].mean(), 0, decimal=5)
        npt.assert_almost_equal(r2.mean(), 1, decimal=5)

    def test_multi_threaded(self):
        model = Model(self.pixel_array, self.x, self.linear_eq,
                      multithread=True)
        model.initial_guess = [0.9, 0.9]
        model.bounds = ([0, 0], [2, 2])
        model.generate_lists()
        popt, error, r2 = fit_image(model)

        assert len(popt) == 2
        assert len(error) == 2

        assert popt[0].shape == (10, 10, 3)
        assert error[0].shape == (10, 10, 3)
        assert r2.shape == (10, 10, 3)

        npt.assert_almost_equal(popt[0].mean(), 1, decimal=5)
        npt.assert_almost_equal(error[0].mean(), 0, decimal=5)
        npt.assert_almost_equal(r2.mean(), 1, decimal=5)
