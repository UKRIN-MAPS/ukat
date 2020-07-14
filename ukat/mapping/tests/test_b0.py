import numpy as np
import pytest
from ukat.mapping.b0 import b0map
import ukat.utils.tools as tools

# Gold Standard = [mean, std, minimum, maximum]
# B0 inputs: {np.arange(200).reshape((10, 10, 2))} and {[4, 7]}
gold_standard = [336.68341708542715, 4.721768995590352e-14,
                 336.6834170854271, 336.68341708542715]

# Create arrays for testing
correct_array = np.arange(200).reshape((10, 10, 2))
one_echo_array = np.arange(100).reshape((10, 10, 1))
multiple_echoes_array = (np.concatenate((correct_array,
                         np.arange(300).reshape((10, 10, 3))), axis=2))
five_dim_array = np.arange(20000).reshape((10, 10, 10, 10, 2))
correct_echo_list = [4, 7]
one_echo_list = [4]
multiple_echo_list = [1, 2, 3, 4, 5]


def test_b0map_values():
    b0_map_calculated = b0map(correct_array, correct_echo_list)
    np.testing.assert_allclose(tools.image_stats(b0_map_calculated),
                               gold_standard, rtol=1e-7, atol=1e-9)


def test_array_input_output_shapes():
    output_array_1 = b0map(correct_array, correct_echo_list)
    output_array_2 = b0map(multiple_echoes_array, multiple_echo_list)
    assert np.shape(output_array_1) == np.shape(output_array_2)
    assert np.shape(output_array_1) == (10, 10)


def test_echo_list_lengths():
    output_array_1 = b0map(correct_array, correct_echo_list)
    output_array_2 = b0map(multiple_echoes_array, multiple_echo_list)
    output_array_3 = b0map(correct_array, multiple_echo_list)
    output_array_4 = b0map(multiple_echoes_array, correct_echo_list)
    assert (output_array_1 == output_array_4).all()
    assert (output_array_2 == output_array_3).all()


def test_b0map_difference():
    difference_result = b0map(np.concatenate((one_echo_array, one_echo_array),
                              axis=2), correct_echo_list)
    assert (difference_result == np.zeros((10, 10))).all()


def test_unwrap_phase_flag():
    unwrapped = b0map(correct_array, correct_echo_list)
    wrapped = b0map(correct_array, correct_echo_list,
                    unwrap=False)
    assert (unwrapped != wrapped).any()


def test_array_ndims():
    with pytest.raises(ValueError):
        b0map(five_dim_array, correct_echo_list)


def test_one_echo_errors():
    with pytest.raises(IndexError):
        b0map(correct_array, one_echo_list)
        b0map(one_echo_array, correct_echo_list)
        b0map(one_echo_array, one_echo_list)


def test_pixel_array_type_assertion():
    # Empty array
    with pytest.raises(ValueError):
        b0map(np.array([]), correct_echo_list)
    with pytest.raises(TypeError):
        # No input argument
        b0map(None, correct_echo_list)
        # Other type
        b0map(list([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]), correct_echo_list)
    # Other type
    with pytest.raises(TypeError):
        b0map("abcdef", correct_echo_list)


def test_echo_list_type_assertion():
    # Empty list
    with pytest.raises(IndexError):
        b0map(correct_array, np.array([]))
    # No input argument
    with pytest.raises(TypeError):
        b0map(correct_array, None)
    # Other types
    with pytest.raises(TypeError):
        b0map(correct_array, 3.2)
        b0map(correct_array, "abcdef")


if __name__ == '__main__':
    pytest.main()
