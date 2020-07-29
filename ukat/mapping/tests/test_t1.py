import numpy as np
import pytest
from ukat.mapping.t1 import magnitude_correct
import ukat.utils.tools as tools

np.random.seed(1)

# Magnitude Correction Testing

real = np.array([-43611, -46086, -19840, -14032, 8654])
imag = np.array([51432, 30621, 5189, 4677, -6265])

# Numpy complex data
comp = real + imag * (0 + 1j)
# "Simple data" uses the last dimension to hold real and imaginary
# components respectively
simple = np.array([real, imag]).T

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
