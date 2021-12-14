import csv
import os
import nibabel as nib
import numpy as np

from segment import Tkv


class Segmentation(nib.Nifti1Image):
    def __init__(self, pixel_array, affine, post_process=True):
        super().__init__(pixel_array, affine)
        self._seg_obj = Tkv(super())
        self._mask = self._seg_obj.get_mask(post_process=post_process)
        self._kidneys = (self._mask > 0.5) * 1
        self._kidneys[:self.shape[0]//2] *= 2
        self.volumes = {'total': self._calculate_volume(self._mask > 0.5),
                        'left': self._calculate_volume(self._kidneys == 1),
                        'right': self._calculate_volume(self._kidneys == 2)}

    def get_mask(self, binary=True):
        if binary:
            mask = self._mask > 0.5
        else:
            mask = self._mask
        return mask

    def get_kidneys(self):
        return self._kidneys

    def get_left_kidney(self):
        left_kidney = (self._kidneys == 1) * 1
        return left_kidney

    def get_right_kidney(self):
        right_kidney = (self._kidneys == 2) * 1
        return right_kidney

    def get_volumes(self):
        return self.volumes

    def save_volumes_csv(self, path):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, self.volumes.keys())
            w.writeheader()
            w.writerow(self.volumes)

    def get_tkv(self):
        return self.volumes['total']

    def get_lkv(self):
        return self.volumes['left']

    def get_rkv(self):
        return self.volumes['right']

    def to_nifti(self, output_directory=os.getcwd(), base_file_name='Output',
                 maps='all'):
        os.makedirs(output_directory, exist_ok=True)
        base_path = os.path.join(output_directory, base_file_name)
        if maps == 'all' or maps == ['all']:
            maps = ['mask', 'left', 'right', 'individual']

        if isinstance(maps, list):
            for result in maps:
                if result == 'mask':
                    mask_nifti = nib.Nifti1Image(self._mask, self.affine)
                    nib.save(mask_nifti, base_path + '_mask.nii.gz')
                elif result == 'left':
                    left_nifti = nib.Nifti1Image(self.get_left_kidney(),
                                                 self.affine)
                    nib.save(left_nifti, base_path + '_left_kidney.nii.gz')
                elif result == 'right':
                    right_nifti = nib.Nifti1Image(self.get_right_kidney(),
                                                 self.affine)
                    nib.save(right_nifti, base_path + '_right_kidney.nii.gz')
                elif result == 'individual':
                    ind_nifti = nib.Nifti1Image(self.get_kidneys(),
                                                 self.affine)
                    nib.save(ind_nifti, base_path +
                             '_individual_kidneys.nii.gz')
        else:
            raise ValueError('No NIFTI file saved. The variable "maps" '
                             'should be "all" or a list of maps from '
                             '"["mask", "left", "right", "individual"]".')

    def _calculate_volume(self, mask):
        mask = mask.astype(bool)
        return np.sum(mask) * np.prod(self.header.get_zooms())
