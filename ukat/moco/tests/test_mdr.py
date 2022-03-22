import os
import shutil
import numpy as np
import numpy.testing as npt
import pytest

from ukat.data import fetch
from ukat.moco.mdr import MotionCorrection
from ukat.utils import arraystats


class TestMotionCorrection:
    pixel_array, affine, bvals, bvecs = fetch.dwi_philips()
    # To avoid long processing times, we will process the DWI data only.
    # This will affect code coverage metrics
    # Single-slice
    # Maybe use a custom elastix parameter?
    image_dwi_1 = np.squeeze(pixel_array[:, :, 2, :]) # can use T1 phantom instead
    # 2 Central slices
    image_dwi_2 = pixel_array[:, :, 4:6, :]
    dwi_input_list = [affine, bvals]
    # Create MotionCorrection instance for each dataset
    #registration_slice = MotionCorrection(image_dwi_1, affine, 'DWI_Moco',
                                          #dwi_input_list)
    #registration_volume = MotionCorrection(image_dwi_2, affine, 'DWI_Moco',
                                           #dwi_input_list)


    def test_run_registration(self):
        #self.registration_slice.mdr_results != []
        #self.registration_volume.mdr_results != []
        self.__class__.new_test_variable = 1000101

    def test_results_post_run(self):
        print(self.new_test_variable)
        assert len(self.registration_slice.mdr_results) == 5
        assert len(self.registration_volume.mdr_results) == 2
        assert len(self.registration_volume.mdr_results[0]) == 5
        # Test that the first element of the results is np.ndarray
        # for 3D input images
        assert isinstance(self.registration_slice.mdr_results[0], np.ndarray) == True
        # Test that the first element of the results is a list
        # for 4D input images
        assert isinstance(self.registration_volume.mdr_results[0], list) == True
        
    # Test individual outputs in terms of stats
    def test_coregistered(self):
        expected_dwi_1 = []
        expected_dwi_2 = []
        coregistered_dwi_1 = self.registration_slice.get_coregistered()
        coregistered_dwi_2 = self.registration_volume.get_coregistered()
        dwi_1_stats = arraystats.ArrayStats(coregistered_dwi_1).calculate()
        dwi_2_stats = arraystats.ArrayStats(coregistered_dwi_2).calculate()
        print([dwi_2_stats["mean"]["4D"], dwi_2_stats["std"]["4D"], dwi_2_stats["min"]["4D"], dwi_2_stats["max"]["4D"]])
        # [15978.658318769889, 20792.03407953149, -6.882346781367232e-08, 383545.53125]
        print([dwi_1_stats["mean"]["3D"], dwi_1_stats["std"]["3D"], dwi_1_stats["min"]["3D"], dwi_1_stats["max"]["3D"]])
        # [14463.805083112677, 16900.571491925268, -2.0136616285526543e-07, 361176.40625]
        npt.assert_allclose([dwi_2_stats["mean"]["4D"], dwi_2_stats["std"]["4D"],
                             dwi_2_stats["min"]["4D"], dwi_2_stats["max"]["4D"]],
                             expected_dwi_2, rtol=1e-6, atol=1e-4)
        npt.assert_allclose([dwi_1_stats["mean"]["3D"], dwi_1_stats["std"]["3D"],
                             dwi_1_stats["min"]["3D"], dwi_1_stats["max"]["3D"]],
                             expected_dwi_1, rtol=1e-6, atol=1e-4)

    def test_fitted(self):
        expected_dwi_1 = []
        expected_dwi_2 = []
        fitted_dwi_1 = self.registration_slice.get_fitted()
        fitted_dwi_2 = self.registration_volume.get_fitted()
        dwi_1_stats = arraystats.ArrayStats(fitted_dwi_1).calculate()
        dwi_2_stats = arraystats.ArrayStats(fitted_dwi_2).calculate()
        print([dwi_2_stats["mean"]["4D"], dwi_2_stats["std"]["4D"], dwi_2_stats["min"]["4D"], dwi_2_stats["max"]["4D"]])
        # [17603.118467089615, 22408.50537554064, -6.882346781367232e-08, 365905.34375]
        print([dwi_1_stats["mean"]["3D"], dwi_1_stats["std"]["3D"], dwi_1_stats["min"]["3D"], dwi_1_stats["max"]["3D"]])
        # [15749.459371950186, 17950.82478985577, -2.0136616285526543e-07, 337065.5]
        npt.assert_allclose([dwi_2_stats["mean"]["4D"], dwi_2_stats["std"]["4D"],
                             dwi_2_stats["min"]["4D"], dwi_2_stats["max"]["4D"]],
                             expected_dwi_2, rtol=1e-6, atol=1e-4)
        npt.assert_allclose([dwi_1_stats["mean"]["3D"], dwi_1_stats["std"]["3D"],
                             dwi_1_stats["min"]["3D"], dwi_1_stats["max"]["3D"]],
                             expected_dwi_1, rtol=1e-6, atol=1e-4)

    def test_deformation_field(self):
        deformation_dwi_1 = self.registration_slice.get_deformation_field()
        deformation_dwi_2 = self.registration_volume.get_deformation_field()
        print(np.shape(deformation_dwi_1))
        print(np.shape(deformation_dwi_2))
        assert 1 == 1
    
    def test_output_parameters(self):
        expected_dwi_1 = []
        expected_dwi_2 = []
        parameter_dwi_1 = self.registration_slice.get_output_parameters()
        parameter_dwi_2 = self.registration_volume.get_output_parameters()
        dwi_1_stats = arraystats.ArrayStats(parameter_dwi_1).calculate()
        dwi_2_stats = arraystats.ArrayStats(parameter_dwi_2).calculate()
        print([dwi_2_stats["mean"]["3D"], dwi_2_stats["std"]["3D"], dwi_2_stats["min"]["3D"], dwi_2_stats["max"]["3D"]])
        print([dwi_1_stats["mean"]["3D"], dwi_1_stats["std"]["3D"], dwi_1_stats["min"]["3D"], dwi_1_stats["max"]["3D"]])
        npt.assert_allclose([dwi_2_stats["mean"]["3D"], dwi_2_stats["std"]["3D"],
                             dwi_2_stats["min"]["3D"], dwi_2_stats["max"]["3D"]],
                             expected_dwi_2, rtol=1e-6, atol=1e-4)
        npt.assert_allclose([dwi_1_stats["mean"]["3D"], dwi_1_stats["std"]["3D"],
                             dwi_1_stats["min"]["3D"], dwi_1_stats["max"]["3D"]],
                             expected_dwi_1, rtol=1e-6, atol=1e-4)

    def test_improvements(self):
        improvements_dwi_1 = self.registration_slice.get_improvements()
        improvements_dwi_2 = self.registration_volume.get_improvements()
        print(improvements_dwi_1)
        print(improvements_dwi_2)
        assert 1 == 1
        
    def test_to_nifti(self):
        os.makedirs('test_output', exist_ok=True)

        # Check all is saved for T1 Moco.
        self.registration_slice.to_nifti(output_directory='test_output',
                                         base_file_name='test_dwi_1',
                                         maps='all')
        output_files = os.listdir('test_output')
        assert len(output_files) == 6
        assert 'test_dwi_1_mask.nii.gz' in output_files
        assert 'test_dwi_1_original.nii.gz' in output_files
        assert 'test_dwi_1_coregistered.nii.gz' in output_files
        assert 'test_dwi_1_fitted.nii.gz' in output_files
        assert 'test_dwi_1_deformation_field.nii.gz' in output_files
        assert 'test_dwi_1_parameters.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        with pytest.raises(ValueError):
            self.registration_slice.to_nifti(output_directory='test_output',
                                             base_file_name='test_dwi_1',
                                             maps='not_an_option')
                            
        # Check coregistered and fiited are saved for DWI Moco.
        self.registration_volume.to_nifti(output_directory='test_output',
                                          base_file_name='test_dwi_2',
                                          maps=['original', 'coregistered'])
        output_files = os.listdir('test_output')
        assert len(output_files) == 2
        assert 'test_dwi_2_original.nii.gz' in output_files
        assert 'test_dwi_2_coregistered.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        with pytest.raises(ValueError):
            self.registration_volume.to_nifti(output_directory='test_output',
                                              base_file_name='test_dwi_2',
                                              maps='not_an_option')

        # Delete 'test_output' folder
        shutil.rmtree('test_output')


if os.path.exists('test_output'):
    shutil.rmtree('test_output')
