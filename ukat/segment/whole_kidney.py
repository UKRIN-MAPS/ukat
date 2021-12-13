import nibabel as nib
import numpy as np


class Segmentation(nib.Nifti1Image):
    def __init__(self, pixel_array, affine, mask):
        super().__init__(pixel_array, affine)
        self._mask = mask

    def get_mask(self):
        return self._mask

    def tkv(self):
        return np.sum(self._mask) * np.prod(self.header.get_zooms())
