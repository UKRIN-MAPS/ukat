import numpy as np
import numpy.testing as npt
import pytest
from ukat.mapping.t1 import T1, magnitude_correct, two_param_eq, \
    two_param_abs_eq, three_param_eq, three_param_abs_eq
import ukat.utils.tools as tools


class TestT1:
    t1 = 1000
    m0 = 5000
    eff = 1.8
    t = np.linspace(200, 1000, 9)

    # The ideal signal produced by the equation m0 * (1 - 2 * exp(-t / t1))
    # where m0 = 5000 and t1 = 1000 at 9 t between 200 and 1000 ms
    correct_signal_two_param = np.array([-3187.30753078, -2408.18220682,
                                         -1703.20046036, -1065.30659713,
                                         -488.11636094, 34.14696209,
                                         506.71035883, 934.30340259,
                                         1321.20558829])
    # The idea signal produced by the equation M0 * (1 - eff * exp(-t /
    # T1)) where M0 = 5000, eff = 1.8 and T1 = 1000 at 9 t between 200
    # and 1000 ms
    correct_signal_three_param = np.array([-2368.5767777, -1667.36398614,
                                           -1032.88041432, -458.77593741,
                                           60.69527515, 530.73226588,
                                           956.03932295, 1340.87306233,
                                           1689.08502946])

    def test_two_param_eq(self):
        # Without abs
        signal = two_param_eq(self.t, self.t1, self.m0)
        npt.assert_allclose(signal, self.correct_signal_two_param,
                            rtol=1e-6, atol=1e-8)
        # With abs
        signal = two_param_abs_eq(self.t, self.t1, self.m0)
        npt.assert_allclose(signal, np.abs(self.correct_signal_two_param),
                            rtol=1e-6, atol=1e-8)

    def test_three_param_eq(self):
        # Without abs
        signal = three_param_eq(self.t, self.t1, self.m0, self.eff)
        npt.assert_allclose(signal, self.correct_signal_three_param,
                            rtol=1e-6, atol=1e-8)
        # With abs
        signal = three_param_abs_eq(self.t, self.t1, self.m0, self.eff)
        npt.assert_allclose(signal, np.abs(self.correct_signal_three_param),
                            rtol=1e-6, atol=1e-8)

    def test_two_param_fit(self):
        # Make the signal into a 4D array
        signal_array = np.tile(self.correct_signal_two_param, (10, 10, 3, 1))

        # Multithread
        mapper = T1(signal_array, self.t, multithread=True)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t1_map.mean() - self.t1 < 0.1
        assert mapper.m0_map.mean() - self.m0 < 0.1

        # Single Threaded
        mapper = T1(signal_array, self.t, multithread=False)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t1_map.mean() - self.t1 < 0.1
        assert mapper.m0_map.mean() - self.m0 < 0.1

        # Fail to fit
        mapper = T1(signal_array[..., ::-1], self.t, multithread=True)
        assert mapper.shape == signal_array.shape[:-1]
        # Voxels that fail to fit are set to zero
        assert mapper.t1_map.mean() == 0.0

    def test_three_param_fit(self):
        # Make the signal into a 4D array
        signal_array = np.tile(self.correct_signal_three_param, (10, 10, 3, 1))

        # Multithread
        mapper = T1(signal_array, self.t, parameters=3, multithread=True)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t1_map.mean() - self.t1 < 0.1
        assert mapper.m0_map.mean() - self.m0 < 0.1
        assert mapper.eff_map.mean() - self.eff < 0.05

        # Single Threaded
        mapper = T1(signal_array, self.t, parameters=3, multithread=False)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t1_map.mean() - self.t1 < 0.1
        assert mapper.m0_map.mean() - self.m0 < 0.1
        assert mapper.eff_map.mean() - self.eff < 0.05

        # Fail to fit
        mapper = T1(signal_array[..., ::-1], self.t,
                    parameters=3, multithread=True)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t1_map.mean() == 0.0

    def test_mask(self):
        signal_array = np.tile(self.correct_signal_two_param, (10, 10, 3, 1))

        # Bool mask
        mask = np.ones(signal_array.shape[:-1], dtype=bool)
        mask[:5, :, :] = False
        mapper = T1(signal_array, self.t, mask=mask)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t1_map[5:, :, :].mean() - self.t1 < 0.1
        assert mapper.t1_map[:5, :, :].mean() < 0.1

        # Int mask
        mask = np.ones(signal_array.shape[:-1])
        mask[:5, :, :] = 0
        mapper = T1(signal_array, self.t, mask=mask)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t1_map[5:, :, :].mean() - self.t1 < 0.1
        assert mapper.t1_map[:5, :, :].mean() < 0.1

    def test_missmatched_raw_data_and_inversion_lengths(self):

        with pytest.raises(AssertionError):
            mapper = T1(pixel_array=np.zeros((5, 5, 4)),
                        inversion_list=np.linspace(0, 2000, 5))

        with pytest.raises(AssertionError):
            mapper = T1(pixel_array=np.zeros((5, 5, 5)),
                        inversion_list=np.linspace(0, 2000, 4))

    def test_parameters(self):

        # One parameter fit
        with pytest.raises(ValueError):
            mapper = T1(pixel_array=np.zeros((5, 5, 5)),
                        inversion_list=np.linspace(0, 2000, 5),
                        parameters=1)

        # Four parameter fit
        with pytest.raises(ValueError):
            mapper = T1(pixel_array=np.zeros((5, 5, 5)),
                        inversion_list=np.linspace(0, 2000, 5),
                        parameters=4)


class TestMagnitudeCorrect:

    real = np.array([-43611, -46086, -19840, -14032, 8654])
    imag = np.array([51432, 30621, 5189, 4677, -6265])

    # Numpy complex data
    comp = real + imag * (0 + 1j)
    # "Simple data" uses the last dimension to hold real and imaginary
    # components respectively
    simple = np.array([real, imag]).T

    correct_array = np.array([-67432.70678981, -55331.41094351,
                              -20507.34797579, -14790.92130329,
                              10683.72318061])

    def test_complex_conversion_shape(self):

        # Has no effect on already complex data
        corrected = magnitude_correct(self.comp)
        assert corrected.shape == (5,)

        # Converts the last dimension to complex data
        corrected = magnitude_correct(self.simple)
        assert corrected.shape == (5,)

        # Raise error if not complex data but last dimension doesn't have
        # length two i.e. isn't real and imag
        with pytest.raises(ValueError):
            corrected = magnitude_correct(self.simple[:, 0])

    def test_input_dimensions(self):

        # Tile existing data to increase dimensions
        # Comp tested up to 4D i.e. [x, y, z, TI]
        comp_2d = np.tile(self.comp, (4, 1))
        comp_3d = np.tile(self.comp, (4, 4, 1))
        comp_4d = np.tile(self.comp, (4, 4, 4, 1))
        # Simple tested up to 5D i.e. [x, y, z, TI, re/im]
        simp_3d = np.tile(self.simple, (4, 1, 1))
        simp_4d = np.tile(self.simple, (4, 4, 1, 1))
        simp_5d = np.tile(self.simple, (4, 4, 4, 1, 1))

        corrected = magnitude_correct(comp_2d)
        assert corrected.shape == (4, 5)
        npt.assert_allclose(corrected,
                            np.tile(self.correct_array, (4, 1)),
                            rtol=1e-9, atol=1e-9)

        corrected = magnitude_correct(comp_3d)
        assert corrected.shape == (4, 4, 5)
        npt.assert_allclose(corrected,
                            np.tile(self.correct_array, (4, 4, 1)),
                            rtol=1e-9, atol=1e-9)

        corrected = magnitude_correct(comp_4d)
        assert corrected.shape == (4, 4, 4, 5)
        npt.assert_allclose(corrected,
                            np.tile(self.correct_array, (4, 4, 4, 1)),
                            rtol=1e-9, atol=1e-9)

        corrected = magnitude_correct(simp_3d)
        assert corrected.shape == (4, 5)
        npt.assert_allclose(corrected,
                            np.tile(self.correct_array, (4, 1)),
                            rtol=1e-9, atol=1e-9)

        corrected = magnitude_correct(simp_4d)
        assert corrected.shape == (4, 4, 5)
        npt.assert_allclose(corrected,
                            np.tile(self.correct_array, (4, 4, 1)),
                            rtol=1e-9, atol=1e-9)

        corrected = magnitude_correct(simp_5d)
        assert corrected.shape == (4, 4, 4, 5)
        npt.assert_allclose(corrected,
                            np.tile(self.correct_array, (4, 4, 4, 1)),
                            rtol=1e-9, atol=1e-9)
