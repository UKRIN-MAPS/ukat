import os
import shutil
import numpy as np
import pandas as pd
import numpy.testing as npt
import pytest
import itk

from ukat.data import fetch
from ukat.moco.mdr import MotionCorrection
from ukat.mapping.t1 import magnitude_correct
from ukat.utils.tools import convert_to_pi_range
from ukat.utils import arraystats


class TestMotionCorrection:
    magnitude, phase, affine_t1, ti, tss = fetch.t1_philips(2)
    ti = np.array(ti) * 1000  # convert TIs to ms
    pixel_array, affine_dwi, bvals, bvecs = fetch.dwi_philips()
    # Preprocess T1 images
    phase = convert_to_pi_range(phase)
    complex_data = magnitude * (np.cos(phase) + 1j * np.sin(phase))
    magnitude_corrected = np.squeeze(magnitude_correct(complex_data))
    # Single-slice + T1/DWI Model Fitting
    image_t1_slice = np.nan_to_num(np.squeeze(magnitude_corrected[:, :, 2, :]))
    t1_slice_input = [affine_t1, ti, 0, None]
    image_dwi = np.nan_to_num(pixel_array[:, :, 5, :])
    dwi_input = [affine_dwi, bvals]
    # Multiple slices + Constant Model Fitting
    image_t1 = np.nan_to_num(magnitude_corrected)
    t1_input = [affine_t1, ti, tss]

    def test_run_mdr(self):
        self.__class__.mdr_t1_slice = MotionCorrection(self.image_t1_slice,
                                                       self.affine_t1,
                                                       'T1_Moco',
                                                       self.t1_slice_input,
                                                       log=False)
        self.__class__.mdr_dwi = MotionCorrection(self.image_dwi,
                                                  self.affine_dwi, 'DWI_Moco',
                                                  self.dwi_input,
                                                  log=False)
        self.__class__.mdr_t1 = MotionCorrection(self.image_t1,
                                                 self.affine_t1,
                                                 'No Fit',
                                                 self.t1_input,
                                                 log=False)

    # Test individual outputs in terms of stats
    def test_coregistered_and_difference(self):
        # It's challenging to set metrics and determine what is a successful
        # motion correction result. One option for the future is to create 
        # digital phantoms for `ukat`. At this moment in time, these tests
        # will check the difference image calculation and inspect if it's
        # different to zero to prove that the MDR did something to the input.
        coregistered_t1_slice = self.mdr_t1_slice.get_coregistered()
        coregistered_dwi = self.mdr_dwi.get_coregistered()
        coregistered_t1 = self.mdr_t1.get_coregistered()
        difference_t1_slice = self.mdr_t1_slice.get_coregistered()
        difference_dwi = self.mdr_dwi.get_coregistered()
        difference_t1 = self.mdr_t1.get_coregistered()

        # The following asserts check if the difference image is different to 0
        assert np.nanmedian(difference_t1_slice) != 0
        assert np.nanmedian(difference_dwi) != 0
        assert np.nanmedian(difference_t1) != 0
        assert np.unique(difference_t1_slice) != [0.0]
        assert np.unique(difference_dwi) != [0.0]
        assert np.unique(difference_t1) != [0.0]

    def test_model_fit(self):
        # Regardless of the co-registration result, it's expected that the
        # model_fit output is consistently the same.
        fitted_t1_slice = self.mdr_t1_slice.get_model_fit()
        fitted_dwi = self.mdr_dwi.get_model_fit()
        fitted_t1 = self.mdr_t1.get_model_fit()
        t1_slice_stats = arraystats.ArrayStats(fitted_t1_slice).calculate()
        dwi_stats = arraystats.ArrayStats(fitted_dwi).calculate()
        t1_stats = arraystats.ArrayStats(fitted_t1).calculate()
        t1_slice_expected = [56.368349731354584, 154.9382840328853,
                             -1477.105499971348, 1349.2699472113275]
        dwi_expected = [16589.810320854365, 22772.03431902955,
                        -6.882346781367232e-08, 364102.84375]
        t1_expected = [59.375521241962176, 115.06861216994638,
                       -722.2862304051718, 3007.8351923624673]
        npt.assert_allclose([t1_slice_stats["mean"]["3D"], t1_slice_stats["std"]["3D"],
                             t1_slice_stats["min"]["3D"], t1_slice_stats["max"]["3D"]],
                             t1_slice_expected, rtol=1e-6, atol=1e-4)
        npt.assert_allclose([dwi_stats["mean"]["3D"], dwi_stats["std"]["3D"],
                             dwi_stats["min"]["3D"], dwi_stats["max"]["3D"]],
                             dwi_expected, rtol=1e-6, atol=1e-4)
        npt.assert_allclose([t1_stats["mean"]["4D"], t1_stats["std"]["4D"],
                             t1_stats["min"]["4D"], t1_stats["max"]["4D"]],
                             t1_expected, rtol=1e-6, atol=1e-4)

    def test_deformation_field(self):
        deformation_t1_slice = self.mdr_t1_slice.get_deformation_field()
        deformation_dwi = self.mdr_dwi.get_deformation_field()
        deformation_t1 = self.mdr_t1.get_deformation_field()
        print(np.shape(deformation_t1_slice))
        print(np.shape(deformation_dwi))
        print(np.shape(deformation_t1))
        assert np.shape(deformation_t1_slice) == (128, 128, 2, 18)
        assert np.shape(deformation_dwi) == (128, 128, 2, 79)
        assert np.shape(deformation_t1) == (128, 128, 5, 2, 18)
    
    def test_parameters(self):
        # Regardless of the co-registration result, it's expected that the
        # parameters outputs are consistently the same.
        m0_t1_slice = self.mdr_t1_slice.get_parameters()[1]
        adc_dwi = self.mdr_dwi.get_parameters()[0]
        t1map_t1 = self.mdr_t1.get_parameters()[0]
        m0_t1_slice_stats = arraystats.ArrayStats(m0_t1_slice).calculate()
        adc_dwi_stats = arraystats.ArrayStats(adc_dwi).calculate()
        t1map_stats = arraystats.ArrayStats(t1map_t1).calculate()
        print([m0_t1_slice_stats["mean"]["2D"], m0_t1_slice_stats["std"]["2D"], m0_t1_slice_stats["min"]["2D"], m0_t1_slice_stats["max"]["2D"]])
        m0_t1_slice_expected = []
        print([adc_dwi_stats["mean"]["2D"], adc_dwi_stats["std"]["2D"], adc_dwi_stats["min"]["2D"], adc_dwi_stats["max"]["2D"]])
        adc_dwi_expected = []
        print([t1map_stats["mean"]["3D"], t1map_stats["std"]["3D"], t1map_stats["min"]["3D"], t1map_stats["max"]["3D"]])
        t1map_expected = []
        npt.assert_allclose([m0_t1_slice_stats["mean"]["3D"], m0_t1_slice_stats["std"]["3D"],
                             m0_t1_slice_stats["min"]["3D"], m0_t1_slice_stats["max"]["3D"]],
                             m0_t1_slice_expected, rtol=1e-6, atol=1e-4)
        npt.assert_allclose([adc_dwi_stats["mean"]["3D"], adc_dwi_stats["std"]["3D"],
                             adc_dwi_stats["min"]["3D"], adc_dwi_stats["max"]["3D"]],
                             adc_dwi_expected, rtol=1e-6, atol=1e-4)
        npt.assert_allclose([t1map_stats["mean"]["4D"], t1map_stats["std"]["4D"],
                             t1map_stats["min"]["4D"], t1map_stats["max"]["4D"]],
                             t1map_expected, rtol=1e-6, atol=1e-4)

    def test_improvements(self):
        os.makedirs('test_output', exist_ok=True)
        imprv_dwi = self.mdr_dwi.get_improvements()
        imprv_t1 = self.mdr_t1.get_improvements(output_directory='test_output',
                                                base_file_name='improvements',
                                                export=True)
        output_files = os.listdir('test_output')
        print(len(output_files))
        assert len(output_files) == 5
        assert 'improvements_slice_0.csv' in output_files
        assert 'improvements_slice_1.csv' in output_files
        assert 'improvements_slice_2.csv' in output_files
        assert 'improvements_slice_3.csv' in output_files
        assert 'improvements_slice_4.csv' in output_files
        assert (float(imprv_dwi['Maximum deformation'].iloc[-1]) < \
                self.mdr_dwi.convergence)
        for imprv_slc in imprv_t1:
            assert (float(imprv_slc['Maximum deformation'].iloc[-1]) < \
                    self.mdr_t1.convergence)
        
        # Delete 'test_output' folder
        shutil.rmtree('test_output')
    
    def test_elastix_parameters(self):
        os.makedirs('test_output', exist_ok=True)
        elastix_dwi = self.mdr_dwi.get_elastix_parameters()
        elastix_t1 = (self.mdr_t1.get_elastix_parameters(
                      output_directory='test_output',
                      base_file_name='elastix',
                      export=True))
        output_files = os.listdir('test_output')
        print(len(output_files))
        assert len(output_files) == 1
        assert 'elastix.txt' in output_files
        assert elastix_dwi['Transform'] == ['BSplineTransform']
        assert elastix_t1['Transform'] == ['EulerTransform']
        
        # Delete 'test_output' folder
        shutil.rmtree('test_output')
        
    def test_to_nifti(self):
        os.makedirs('test_output', exist_ok=True)
        # Check all is saved for DWI Moco.
        self.mdr_dwi.to_nifti(output_directory='test_output',
                              base_file_name='test_dwi',
                              maps='all')
        output_files = os.listdir('test_output')
        assert len(output_files) == 7
        assert 'test_dwi_mask.nii.gz' in output_files
        assert 'test_dwi_original.nii.gz' in output_files
        assert 'test_dwi_coregistered.nii.gz' in output_files
        assert 'test_dwi_difference.nii.gz' in output_files
        assert 'test_dwi_model_fit.nii.gz' in output_files
        assert 'test_dwi_deformation_field.nii.gz' in output_files
        assert 'test_dwi_parameters.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        with pytest.raises(ValueError):
            self.mdr_dwi.to_nifti(output_directory='test_output',
                                  base_file_name='test_dwi',
                                  maps='not_an_option')

        # Check coregistered and fitted are saved for T1 Moco.
        self.mdr_t1.to_nifti(output_directory='test_output',
                             base_file_name='test_t1',
                             maps=['model_fit', 'coregistered'])
        output_files = os.listdir('test_output')
        assert len(output_files) == 2
        assert 'test_t1_model_fit.nii.gz' in output_files
        assert 'test_t1_coregistered.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        with pytest.raises(ValueError):
            self.mdr_t1.to_nifti(output_directory='test_output',
                                 base_file_name='test_t1',
                                 maps='not_an_option')

        # Delete 'test_output' folder
        shutil.rmtree('test_output')
    
    def test_to_gif(self):
        os.makedirs('test_output', exist_ok=True)
        # Check all is saved for DWI Moco.
        self.mdr_dwi.to_gif(output_directory='test_output',
                            base_file_name='test_dwi',
                            maps='all')
        output_files = os.listdir('test_output')
        assert len(output_files) == 7
        assert 'test_dwi_mask.gif' in output_files
        assert 'test_dwi_original.gif' in output_files
        assert 'test_dwi_coregistered.gif' in output_files
        assert 'test_dwi_difference.gif' in output_files
        assert 'test_dwi_model_fit.gif' in output_files
        assert 'test_dwi_deformation_field.gif' in output_files
        assert 'test_dwi_parameters.gif' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        with pytest.raises(ValueError):
            self.mdr_dwi.to_gif(output_directory='test_output',
                                base_file_name='test_dwi',
                                maps='not_an_option')

        # Check coregistered and fitted are saved for T1 Moco.
        self.mdr_t1.to_gif(output_directory='test_output',
                           base_file_name='test_t1',
                           maps=['model_fit', 'coregistered'])
        output_files = os.listdir('test_output')
        assert len(output_files) == 2
        assert 'test_t1_model_fit.gif' in output_files
        assert 'test_t1_coregistered.gif' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        with pytest.raises(ValueError):
            self.mdr_t1.to_gif(output_directory='test_output',
                               base_file_name='test_t1',
                               maps='not_an_option')

        # Delete 'test_output' folder
        shutil.rmtree('test_output')


if os.path.exists('test_output'):
    shutil.rmtree('test_output')
