import csv
import os
import nibabel as nib
import numpy as np

from segment import Tkv


class Segmentation(nib.Nifti1Image):
    """
    Attributes
    ----------
    volumes : dict
        Total, left and right kidney volumes in milliliters.
    """

    def __init__(self, pixel_array, affine, post_process=True, binary=True,
                 weights=None):
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
        binary : bool, optional
            Default True.
            If True, the mask returned will be an array of ints, where 1
            represents voxels that are renal tissue and 0 represents
            voxels that are not renal tissue. If False, the mask returned
            will be the probability that each voxel is renal tissue.
        weights : str, optional
            Default 'none'.
            The path to custom neural network weights. If 'none',
            the default, all-purpose, weights will be downloaded from the
            internet and used.
        """

        super().__init__(pixel_array, affine)
        self.pixel_array = pixel_array
        self._nifti = nib.Nifti1Image(self.pixel_array, self.affine)
        self._seg_obj = Tkv(self._nifti)
        self._mask = self._seg_obj.get_mask(post_process=post_process,
                                            binary=binary,
                                            weights_path=weights)
        self._kidneys = (self._mask > 0.5) * 1
        self._kidneys[:self.shape[0]//2] *= 2
        self.volumes = {'total': self._calculate_volume(self._mask > 0.5),
                        'left': self._calculate_volume(self._kidneys == 1),
                        'right': self._calculate_volume(self._kidneys == 2)}

    def get_mask(self):
        """
        Returns the estimated mask from the provided input data.

        Returns
        -------
        mask : np.ndarray
            The estimated mask.
        """
        return self._mask

    def get_kidneys(self):
        """
        Returns a mask where 0 represents voxels that are not renal tissue,
        1 represents voxels that are the left kidney and 2 represents voxels
        that are the right kidney.

        Returns
        -------
        mask : np.ndarray
            Mask with each kidney represented by a different int.
        """
        return self._kidneys

    def get_left_kidney(self):
        """
        Returns a binary mask where True is the left kidney.

        Returns
        -------
        left_kidney : np.ndarray
            Binary mask of left kidney.
        """
        left_kidney = (self._kidneys == 1) * 1
        return left_kidney

    def get_right_kidney(self):
        """
        Returns a binary mask where True is the right kidney.

        Returns
        -------
        right_kidney : np.ndarray
            Binary mask of right kidney.
        """
        right_kidney = (self._kidneys == 2) * 1
        return right_kidney

    def get_volumes(self):
        """
        Return the volume of each kidney and total kidney volume in
        milliliters.

        Returns
        -------
        volumes : dict
            Total, left and right kidney volumes in milliliters.
        """
        return self.volumes

    def save_volumes_csv(self, path):
        """
        Save the total, left and right kidney volumes to a csv file.

        Parameters
        ----------
        path : str
            Path to the desired csv file.
        """
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, self.volumes.keys())
            w.writeheader()
            w.writerow(self.volumes)

    def get_tkv(self):
        """
        Get the total kidney volume in milliliters.
        Returns
        -------
        tkv : float
            Total kidney volume
        """
        return self.volumes['total']

    def get_lkv(self):
        """
        Get the left kidney volume in milliliters.
        Returns
        -------
        lkv : float
            Left kidney volume
        """
        return self.volumes['left']

    def get_rkv(self):
        """
        Get the right kidney volume in milliliters.
        Returns
        -------
        rkv : float
            Right kidney volume
        """
        return self.volumes['right']

    def to_nifti(self, output_directory=os.getcwd(), base_file_name='Output',
                 maps='all'):
        """Exports masks to NIFTI.

        Parameters
        ----------
        output_directory : string, optional
            Path to the folder where the NIFTI files will be saved.
        base_file_name : string, optional
            Filename of the resulting NIFTI. This code appends the extension.
            Eg., base_file_name = 'Output' will result in 'Output.nii.gz'.
        maps : list or 'all', optional
            List of maps to save to NIFTI. This should either the string "all"
            or a list of maps from ["mask", "left", "right", "individual"]
        """
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
        mask = mask > 0.5
        return np.sum(mask) * np.prod(self.header.get_zooms()) / 1000
