import os
import numpy as np
import nibabel as nib


class MTR:
    """
    Generates a Magnetisation Transfer Ratio (MTR) map from a series of volumes
    collected with 2 different magnetisation transfer values (ON and OFF).

    Attributes
    ----------
    mtr_map : np.ndarray
        The estimated magnetisation transfer ratio values.
        Each value is a float between 0 and 1.
    shape : tuple
        The shape of the MTR map.
    mt_on : np.ndarray
        The array corresponding to the magnetisation transfer value ON.
    mt_off : np.ndarray
        The array corresponding to the magnetisation transfer value OFF.
    mask : np.ndarray
        A boolean mask of the voxels to fit.
    """

    def __init__(self, pixel_array, affine, mask=None):
        """Initialise a MTR class instance.

        Parameters
        ----------
        pixel_array : np.ndarray
            A 4D/3D array containing the image at each magnetisation transfer
            value i.e. the dimensions of the array are
            [x, y, 2] or [x, y, z, 2].
        affine : np.ndarray
            A matrix giving the relationship between voxel coordinates and
            world coordinates.
        mask : np.ndarray, optional
            A boolean mask of the voxels to fit. Should be the shape of the
            desired MTR map rather than the raw data i.e. omit the echo times
            dimension.
        """
        if pixel_array.shape[-1] == 2:
            self.pixel_array = pixel_array
            self.shape = pixel_array.shape[:-1]
            self.affine = affine
            # Generate a mask if there isn't one specified
            if mask is None:
                self.mask = np.ones(self.shape, dtype=bool)
            else:
                self.mask = mask
            # The assumption is that MT_ON comes first in `pixel_array`
            self.mt_on = self.pixel_array[..., 0] * self.mask
            # The assumption is that MT_OFF comes second in `pixel_array`
            self.mt_off = self.pixel_array[..., 1] * self.mask
            # Magnetisation Transfer Ratio calculation
            self.mtr_map = np.nan_to_num(((self.mt_off - self.mt_on) / 
                                           self.mt_off), posinf=0, neginf=0)
        else:
            raise ValueError('The input should contain 2 mt values (ON / OFF).'
                             'The last dimension of the input pixel_array must'
                             'be 2.')


    def to_nifti(self, output_directory=os.getcwd(), base_file_name='Output',
                 maps='all'):
        """Exports some of the MTR class attributes to NIFTI.

        Parameters
        ----------
        output_directory : string, optional
            Path to the folder where the NIFTI files will be saved.
        base_file_name : string, optional
            Filename of the resulting NIFTI. This code appends the extension.
            Eg., base_file_name = 'Output' will result in 'Output.nii.gz'.
        maps : list or 'all', optional
            List of maps to save to NIFTI. This should either the string "all"
            or a list of maps from ["mtr_map", "mt_on", "mt_off", "mask"].
        """
        os.makedirs(output_directory, exist_ok=True)
        base_path = os.path.join(output_directory, base_file_name)
        if maps == 'all' or maps == ['all']:
            maps = ['mtr_map', 'mt_on', 'mt_off', 'mask']
        if isinstance(maps, list):
            for result in maps:
                if result == 'mtr' or result == 'mtr_map':
                    mtr_nifti = nib.Nifti1Image(self.mtr_map,
                                                affine=self.affine)
                    nib.save(mtr_nifti, base_path + '_mtr_map.nii.gz')
                elif result == 'mt_on':
                    mt_on_nifti = nib.Nifti1Image(self.mt_on,
                                                 affine=self.affine)
                    nib.save(mt_on_nifti, base_path + '_mt_on.nii.gz')
                elif result == 'mt_off':
                    mt_off_nifti = nib.Nifti1Image(self.mt_off,
                                                   affine=self.affine)
                    nib.save(mt_off_nifti, base_path + '_mt_off.nii.gz')
                elif result == 'mask':
                    mask_nifti = nib.Nifti1Image(self.mask.astype(int),
                                                 affine=self.affine)
                    nib.save(mask_nifti, base_path + '_mask.nii.gz')
        else:
            raise ValueError('No NIFTI file saved. The variable "maps" '
                             'should be "all" or a list of maps from '
                             '"["mtr_map", "mt_on", "mt_off", "mask"]".')

        return
