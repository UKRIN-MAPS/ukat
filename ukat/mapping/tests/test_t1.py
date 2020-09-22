import numpy as np
import numpy.testing as npt
import pytest
from ukat.mapping.t1 import T1, magnitude_correct
import ukat.utils.tools as tools

# T1 Class Testing

# TODO e2e test with scanner data


def test_two_param_eq():
    correct_signal = np.array([-3187.30753078, -2408.18220682, -1703.20046036,
                               -1065.30659713, -488.11636094,    34.14696209,
                               506.71035883,   934.30340259, 1321.20558829])
    t = np.linspace(200, 1000, 9)
    t1 = 1000
    m0 = 5000
    # Without abs
    signal = T1.__two_param_eq__(t, t1, m0)
    npt.assert_allclose(signal, correct_signal, rtol=1e-6, atol=1e-8)
    # With abs
    signal = T1.__two_param_abs_eq__(t, t1, m0)
    npt.assert_allclose(signal, np.abs(correct_signal), rtol=1e-6, atol=1e-8)


def test_three_param_eq():
    correct_signal = np.array([-2368.5767777, -1667.36398614, -1032.88041432,
                               -458.77593741, 60.69527515, 530.73226588,
                               956.03932295, 1340.87306233, 1689.08502946])
    t = np.linspace(200, 1000, 9)
    t1 = 1000
    m0 = 5000
    eff = 1.8
    # Without abs
    signal = T1.__three_param_eq__(t, t1, m0, eff)
    npt.assert_allclose(signal, correct_signal, rtol=1e-6, atol=1e-8)
    # With abs
    signal = T1.__three_param_abs_eq__(t, t1, m0, eff)
    npt.assert_allclose(signal, np.abs(correct_signal), rtol=1e-6, atol=1e-8)


def test_two_param_fit():
    correct_signal = np.array([-3187.30753078, -2408.18220682, -1703.20046036,
                               -1065.30659713, -488.11636094, 34.14696209,
                               506.71035883, 934.30340259, 1321.20558829])
    signal_array = np.tile(correct_signal, (10, 10, 3, 1))
    ti = np.linspace(200, 1000, 9)
    # Multithread
    mapper = T1(signal_array, ti, multithread=True)
    assert mapper.shape == signal_array.shape[:-1]
    assert mapper.t1_map.mean() - 1000 < 0.1
    assert mapper.m0_map.mean() - 5000 < 0.1

    # Single Threaded
    mapper = T1(signal_array, ti, multithread=False)
    assert mapper.shape == signal_array.shape[:-1]
    assert mapper.t1_map.mean() - 1000 < 0.1
    assert mapper.m0_map.mean() - 5000 < 0.1


def test_three_param_fit():
    correct_signal = np.array([-2368.5767777, -1667.36398614, -1032.88041432,
                               -458.77593741, 60.69527515, 530.73226588,
                               956.03932295, 1340.87306233, 1689.08502946])
    signal_array = np.tile(correct_signal, (10, 10, 3, 1))
    ti = np.linspace(200, 1000, 9)
    # Multithread
    mapper = T1(signal_array, ti, parameters=3, multithread=True)
    assert mapper.shape == signal_array.shape[:-1]
    assert mapper.t1_map.mean() - 1000 < 0.1
    assert mapper.m0_map.mean() - 5000 < 0.1
    assert mapper.eff_map.mean() - 1.8 < 0.05

    # Single Threaded
    mapper = T1(signal_array, ti, parameters=3, multithread=False)
    assert mapper.shape == signal_array.shape[:-1]
    assert mapper.t1_map.mean() - 1000 < 0.1
    assert mapper.m0_map.mean() - 5000 < 0.1
    assert mapper.eff_map.mean() - 1.8 < 0.05


def test_missmatched_raw_data_and_inversion_lengths():

    with pytest.raises(AssertionError):
        mapper = T1(pixel_array=np.zeros((5, 5, 4)),
                    inversion_list=np.linspace(0, 2000, 5))

    with pytest.raises(AssertionError):
        mapper = T1(pixel_array=np.zeros((5, 5, 5)),
                    inversion_list=np.linspace(0, 2000, 4))


def test_parameters():

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


# Magnitude Correction Testing

real = np.array([-43611, -46086, -19840, -14032, 8654])
imag = np.array([51432, 30621, 5189, 4677, -6265])

# Numpy complex data
comp = real + imag * (0 + 1j)
# "Simple data" uses the last dimension to hold real and imaginary
# components respectively
simple = np.array([real, imag]).T

correct_array = np.array([-67432.70678981, -55331.41094351, -20507.34797579,
                          -14790.92130329, 10683.72318061])


# TODO e2e test with scanner data


def test_complex_conversion_shape():

    # Has no effect on already complex data
    corrected = magnitude_correct(comp)
    assert corrected.shape == (5,)

    # Converts the last dimension to complex data
    corrected = magnitude_correct(simple)
    assert corrected.shape == (5,)

    # Raise error if not complex data but last dimension doesn't have length
    # two i.e. isn't real and imag
    with pytest.raises(ValueError):
        corrected = magnitude_correct(simple[:, 0])


def test_input_dimensions():
    
    # Tile existing data to increase dimensions
    # Comp tested up to 4D i.e. [x, y, z, TI]
    comp_2d = np.tile(comp, (4, 1))
    comp_3d = np.tile(comp, (4, 4, 1))
    comp_4d = np.tile(comp, (4, 4, 4, 1))
    # Simple tested up to 5D i.e. [x, y, z, TI, re/im]
    simp_3d = np.tile(simple, (4, 1, 1))
    simp_4d = np.tile(simple, (4, 4, 1, 1))
    simp_5d = np.tile(simple, (4, 4, 4, 1, 1))
    
    corrected = magnitude_correct(comp_2d)
    assert corrected.shape == (4, 5)
    npt.assert_allclose(corrected, np.tile(correct_array, (4, 1)),
                        rtol=1e-9, atol=1e-9)

    corrected = magnitude_correct(comp_3d)
    assert corrected.shape == (4, 4, 5)
    npt.assert_allclose(corrected, np.tile(correct_array, (4, 4, 1)),
                        rtol=1e-9, atol=1e-9)

    corrected = magnitude_correct(comp_4d)
    assert corrected.shape == (4, 4, 4, 5)
    npt.assert_allclose(corrected, np.tile(correct_array, (4, 4, 4, 1)),
                        rtol=1e-9, atol=1e-9)

    corrected = magnitude_correct(simp_3d)
    assert corrected.shape == (4, 5)
    npt.assert_allclose(corrected, np.tile(correct_array, (4, 1)),
                        rtol=1e-9, atol=1e-9)

    corrected = magnitude_correct(simp_4d)
    assert corrected.shape == (4, 4, 5)
    npt.assert_allclose(corrected, np.tile(correct_array, (4, 4, 1)),
                        rtol=1e-9, atol=1e-9)

    corrected = magnitude_correct(simp_5d)
    assert corrected.shape == (4, 4, 4, 5)
    npt.assert_allclose(corrected, np.tile(correct_array, (4, 4, 4, 1)),
                        rtol=1e-9, atol=1e-9)
