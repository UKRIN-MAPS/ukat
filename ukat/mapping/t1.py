import numpy as np


class T1(object):
    """Package containing algorithms that calculate parameter maps
    of the MRI scans acquired during the UKRIN-MAPS project.

    Attributes
    ----------
    See parameters of __init__ class

    """

    def __init__(self, pixel_array, inversion_list):
        """Initialise a T1 class instance.

        Parameters
        ----------
        pixel_array : 4D/3D array
            A 4D/3D array containing the signal from each voxel at each
            inversion time i.e. the dimensions of the array are [x, y, z, TI].
        inversion_list : list()
            An array of the inversion times used for the last dimension of the
            raw data.
        """

        self.pixel_array = pixel_array
        self.inversion_list = inversion_list


def magnitude_correct(pixel_array):
    """Sign corrects the magnitude of inversion recovery data using the
    complex component of the signal.

    This function uses the methods of Jerzy Szumowski et al
    (https://doi.org/10.1002/jmri.23705).

    Parameters
    ----------
    pixel_array: ndarray
        Can either be a complex array or have the real and imaginary
        parts of the image as the final dimension e.g. a complex 3D image
        could have the dimensions [x, y, z, ti] where [0, 0, 0, 0] = 1 + 2j
        or the dimensions [x, y, z, ti, type] where [0, 0, 0, 0, 0] = 1 and
        [0, 0, 0, 0, 1] = 2.

    Returns
    -------
    corrected_array : ndarray
        An array of the magnitude intensities with signs corrected.
    """

    # Convert data to a complex array if it isn't already
    if not np.iscomplexobj(pixel_array):
        if pixel_array.shape[-1] == 2:
            pixel_array = pixel_array[..., 0] + pixel_array[..., 1] * (0 + 1j)
        else:
            raise ValueError('Last axis of pixel_array must have length 2')

    pixel_array_prime = np.zeros(pixel_array.shape, dtype=np.complex128)

    for ti in range(pixel_array.shape[-1]):
        pixel_array_prime[..., ti] = (pixel_array[..., ti] *
                                      pixel_array[..., -1].conjugate()) \
                                     / np.abs(pixel_array[..., -1])

    phase_factor = np.imag(np.log(pixel_array_prime / np.abs(pixel_array)))
    phase_offset = np.abs(phase_factor) - (np.pi / 2)
    sign = -(phase_offset / np.abs(phase_offset))
    corrected_array = sign * np.abs(pixel_array)
    return corrected_array
