import os
import shutil
import numpy as np
import numpy.testing as npt
import pytest

from ukat.data import fetch
from ukat.moco.mdr import MotionCorrection
from ukat.utils import arraystats


class TestMotionCorrection:
    image_t1, phase, affine, ti, tss = fetch.t1_philips(1)
    pixel_array, affine, bvals, bvecs = fetch.dwi_philips()
    image_dwi = pixel_array[:, :, 4:6, :]
    t1_input_list = [affine, ti]
    dwi_input_list = [affine, bvals]
    registration_slice = MotionCorrection(np.squeeze(image_t1), affine,
                                          'T1_Moco', t1_input_list)
    registration_volume = MotionCorrection(image_dwi, affine, 'DWI_Moco',
                                           dwi_input_list)

    def test_results_pre_run(self):
        assert self.registration_slice.mdr_results == []
        assert self.registration_volume.mdr_results == []

    def test_run_registration(self):
        self.registration_slice.run()
        self.registration_volume.run()

    def test_results_post_run(self):
        assert len(self.registration_slice.mdr_results) == 5
        assert len(self.registration_volume.mdr_results) == 5
        # Test that the first element of the results is an nd.array
        # for 3D input images
        print(type(self.registration_slice.mdr_results[0]))
        # Test that the first element of the results is a list
        # for 4D input images
        assert isinstance(self.registration_volume.mdr_results[0], list)
        

    # Test individual outputs in terms of stats
    def test_coregistered(self):
        expected_t1 = []
        expected_dwi = []
        coregistered_t1 = self.registration_slice.get_coregistered()
        coregistered_dwi = self.registration_volume.get_coregistered()
        t1_stats = arraystats.ArrayStats(coregistered_t1).calculate()
        dwi_stats = arraystats.ArrayStats(coregistered_dwi).calculate()
        print([dwi_stats["mean"]["3D"], dwi_stats["std"]["3D"], dwi_stats["min"]["3D"], dwi_stats["max"]["3D"]])
        print([t1_stats["mean"]["3D"], t1_stats["std"]["3D"], t1_stats["min"]["3D"], t1_stats["max"]["3D"]])
        print([t1_stats["mean"]["2D"], t1_stats["std"]["2D"], t1_stats["min"]["2D"], t1_stats["max"]["2D"]])
        npt.assert_allclose([dwi_stats["mean"]["3D"], dwi_stats["std"]["3D"],
                             dwi_stats["min"]["3D"], dwi_stats["max"]["3D"]],
                             expected_dwi, rtol=1e-6, atol=1e-4)
        npt.assert_allclose([t1_stats["mean"]["3D"], t1_stats["std"]["3D"],
                             t1_stats["min"]["3D"], t1_stats["max"]["3D"]],
                             expected_t1, rtol=1e-6, atol=1e-4)

    def test_fitted(self):
        return

    def test_deformation_field(self):
        return
    
    def test_output_parameters(self):
        return

    def test_improvements(self):
        return
        
    def test_to_nifti(self):
        os.makedirs('test_output', exist_ok=True)

        # Check all is saved for T1 Moco.
        self.registration_slice.to_nifti(output_directory='test_output',
                                         base_file_name='test_t1',
                                         maps='all')
        output_files = os.listdir('test_output')
        assert len(output_files) == 5
        assert 'test_t1_mask.nii.gz' in output_files
        assert 'test_t1_coregistered.nii.gz' in output_files
        assert 'test_t1_fitted.nii.gz' in output_files
        assert 'test_t1_deformation_field.nii.gz' in output_files
        assert 'test_t1_parameters.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        with pytest.raises(ValueError):
            self.registration_slice.to_nifti(output_directory='test_output',
                                             base_file_name='test_t1',
                                             maps='not_an_option')
                            
        # Check coregistered and fiited are saved for DWI Moco.
        self.registration_volume.to_nifti(output_directory='test_output',
                                          base_file_name='test_dwi',
                                          maps=['coregistered', 'fitted'])
        output_files = os.listdir('test_output')
        assert len(output_files) == 2
        assert 'test_dwi_coregistered.nii.gz' in output_files
        assert 'test_dwi_fitted.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        with pytest.raises(ValueError):
            self.registration_volume.to_nifti(output_directory='test_output',
                                              base_file_name='test_dwi',
                                              maps='not_an_option')

        # Delete 'test_output' folder
        shutil.rmtree('test_output')


if os.path.exists('test_output'):
    shutil.rmtree('test_output')
