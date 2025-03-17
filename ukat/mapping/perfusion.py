import nibabel as nib
import mdreg
import os
import warnings

class Perfusion:
    def __init__(self, pixel_array, affine, moco=False):
        warnings.warn('Perfusion is still a work in progress and likely to '
                      'be re-written. Use with caution.')
        self.pixel_array = pixel_array
        self.affine = affine
        self.moco = moco

        if self.moco:
            self.pixel_array, self.deformation_field, _, _ = mdreg.fit(
                self.pixel_array, force_2d=True)
        self.label = self.pixel_array[..., 0::2]
        self.control = self.pixel_array[..., 1::2]
        if self.label.mean() < self.control.mean():
            warnings.warn("The average signal from the label image is less "
                          "than the average signal from the control image. ")

        self.mean_label = self.label.mean(axis=-1)
        self.mean_control = self.control.mean(axis=-1)
        self.perfusion_weighted = self.mean_label - self.mean_control

    def to_nifti(self, output_directory=os.getcwd(), base_file_name='Output',
                 maps='all'):
        os.makedirs(output_directory, exist_ok=True)
        base_path = os.path.join(output_directory, base_file_name)
        if maps == 'all' or maps == ['all']:
            maps = ['mean_label', 'mean_control', 'perfusion_weighted',
                    'deformation_field']
        if isinstance(maps, list):
            for result in maps:
                if result == 'mean_label':
                    nifti = nib.Nifti1Image(self.mean_label, self.affine)
                    nib.save(nifti, f"{base_path}_mean_label.nii.gz")
                elif result == 'mean_control':
                    nifti = nib.Nifti1Image(self.mean_control, self.affine)
                    nib.save(nifti, f"{base_path}_mean_control.nii.gz")
                elif result == 'perfusion_weighted':
                    nifti = nib.Nifti1Image(self.perfusion_weighted, self.affine)
                    nib.save(nifti, f"{base_path}_perfusion_weighted.nii.gz")
                elif self.moco is True and result == 'deformation_field':
                    nifti = nib.Nifti1Image(self.deformation_field, self.affine)
                    nib.save(nifti, f"{base_path}_deformation_field.nii.gz")