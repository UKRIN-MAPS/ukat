import numpy as np
import numpy.testing as npt
import pytest
from ukat.mapping.t1 import magnitude_correct
import ukat.utils.tools as tools

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
