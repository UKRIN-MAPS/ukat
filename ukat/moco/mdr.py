import os
import numpy as np
import nibabel as nib
from MDR import MDR

class Segmentation():
    """
    Attributes
    ----------
    volumes : dict
        Total, left and right kidney volumes in milliliters.
    """

    def __init__(self, pixel_array, affine, post_process=True):
        """Initialise a whole kidney segmentation class instance.
        Parameters
        ----------
        pixel_array : np.ndarray
            Array containing a T2-weighted FSE image.
        affine : np.ndarray
            A matrix giving the relationship between voxel coordinates and
            world coordinates.
        post_process : bool, optional
            Default True
            Keep only the two largest connected volumes in the mask. Note
            this may cause issue with subjects that have more or less than
            two kidneys.
        """

        super().__init__(pixel_array, affine)
        self.pixel_array = pixel_array