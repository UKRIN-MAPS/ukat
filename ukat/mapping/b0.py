import numpy as np
from skimage.restoration import unwrap_phase
from ukat.utils.tools import convert_to_pi_range


class B0:
    """
    Generates a B0 map from a series of volumes collected
    with 2 different echo times.

    Attributes
    ----------
    b0_map : np.ndarray
        The estimated B0 values in Hz
    shape : tuple
        The shape of the B0 map
    n_te : int
        The number of TE used to calculate the map
    echo0 : float
        The first Echo Time in ms
    echo1 : float
        The second Echo Time in ms
    delta_te : float
        The difference between the second and the first Echo Time
    phase0 : np.ndarray
        The phase image corresponding to the first Echo Time
    phase1 : np.ndarray
        The phase image corresponding to the second Echo Time
    phase_difference : np.ndarray
        The difference between the 2 phase images
    """

    def __init__(self, pixel_array, echo_list, mask=None, unwrap=True,
                 wrap_around=False):
        """Initialise a T1 class instance.

        Parameters
        ----------
        pixel_array : np.ndarray
            A 4D/3D array containing the phase image at each
            echo time i.e. the dimensions of the array are
            [x, y, TE] or [x, y, z, TE].
        echo_list : list
            An array of the echo times in ms used for the last dimension of the
            raw data.
        unwrap : boolean, optional
            By default, this script applies the
            scipy phase unwrapping for each phase echo image.
        wrap_around : boolean, optional
            By default, this flag from unwrap_phase is False.
            The algorithm will regard the edges along the corresponding axis
            of the image to be connected and use this connectivity to guide the
            phase unwrapping process.Eg., voxels [0, :, :] are considered to be
            next to voxels [-1, :, :] if wrap_around=True.
        """

        self.pixel_array = pixel_array
        self.shape = pixel_array.shape[:-1]
        self.n_te = pixel_array.shape[-1]
        # Generate a mask if there isn't one specified
        if mask is None:
            self.mask = np.ones(self.shape, dtype=bool)
        else:
            self.mask = mask
        if self.n_te == len(echo_list) and self.n_te == 2:
            # Get the Echo Times
            self.echo0 = echo_list[0]
            self.echo1 = echo_list[1]
            # Calculate DeltaTE in seconds
            self.delta_te = np.absolute(self.echo1 - self.echo0) * 0.001
            # Extract each phase image, mask them and rescale to
            # [-pi, pi] if not in that range already.
            self.phase0 = np.ma.masked_array(
                            convert_to_pi_range(np.squeeze(
                                self.pixel_array[..., 0])), mask=mask)
            self.phase1 = np.ma.masked_array(
                            convert_to_pi_range(np.squeeze(
                                self.pixel_array[..., 1])), mask=mask)
            if unwrap:
                # Unwrap each phase image
                self.phase0 = unwrap_phase(self.phase0,
                                           wrap_around=wrap_around)
                self.phase1 = unwrap_phase(self.phase1,
                                           wrap_around=wrap_around)
            # Phase difference
            self.phase_difference = self.phase1 - self.phase0
            # B0 Map calculation
            self.b0_map = self.phase_difference / (2 * np.pi * self.delta_te)
        else:
            raise ValueError('The input should contain 2 echo times.'
                             'The last dimension of the input pixel_array must'
                             'be 2 and the echo_list must only have 2 values.')
