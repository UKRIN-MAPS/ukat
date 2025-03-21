import os
import tempfile
import shutil
import numpy as np
import nibabel as nib
import pytest
import numpy.testing as npt

from ukat.data import fetch
from ukat.mapping.perfusion import Perfusion
from ukat.utils import arraystats


class TestPerfusion:
    """Setup test data that will be used across multiple tests."""
    data, affine = fetch.asl_philips_fair_1500()

    # Create some simple test data
    label_data = np.ones((10, 10, 10)) * 120
    control_data = np.ones((10, 10, 10)) * 100

    # Interleave label and control
    test_array = np.zeros((10, 10, 10, 6))
    test_array[..., 0::2] = label_data[..., np.newaxis]
    test_array[..., 1::2] = control_data[..., np.newaxis]

    # Expected perfusion weighted result
    expected_pwi = np.ones((10, 10, 10)) * (120 - 100)  # label - control

    # Setup identity affine
    test_affine = np.eye(4)
        
    def test_perfusion_initialization(self):
        """Test that the Perfusion class initializes correctly."""
        # Initialize without motion correction
        perf = Perfusion(self.data, self.affine, moco=False)
        
        # Check basic attributes
        assert perf.pixel_array is not None
        assert perf.affine is not None
        assert perf.pixel_array.shape == self.data.shape
        
        # Check that label and control images are separated correctly
        assert perf.label.shape[-1] == self.data.shape[-1] // 2
        assert perf.control.shape[-1] == self.data.shape[-1] // 2
        
        # Check that mean images are calculated
        assert perf.mean_label.ndim == self.data.ndim - 1
        assert perf.mean_control.ndim == self.data.ndim - 1
        
        # Check perfusion-weighted image
        assert perf.perfusion_weighted.shape == perf.mean_label.shape
        assert np.any(perf.perfusion_weighted != 0)  # Should have some non-zero values

    def test_perfusion_calculation(self):
        """Test that perfusion-weighted images are calculated correctly."""
        # Test with the simpler test array where we know the expected result
        perf = Perfusion(self.test_array, self.test_affine)
        
        # Check perfusion weighted image calculation
        npt.assert_array_almost_equal(
            perf.perfusion_weighted, self.expected_pwi,
            decimal=6, err_msg="Perfusion calculation doesn't match expected result"
        )
        
        # Test with the real data
        gold_standard_pwi = [527.179216, 1866.570273, -7242.041576,
                             43664.132142]
        perf_real = Perfusion(self.data, self.affine)

        pwi_stats = arraystats.ArrayStats(perf_real.perfusion_weighted).calculate()
        npt.assert_allclose([pwi_stats["mean"]["3D"],
                             pwi_stats["std"]["3D"],
                             pwi_stats["min"]["3D"],
                             pwi_stats["max"]["3D"]],
                            gold_standard_pwi, rtol=0.1, atol=1E-3)


    def test_perfusion_with_moco(self):
        """Test perfusion calculation with motion correction."""

        gold_standard_pwi_moco = [646.875117, 2613.3713, -17243.585938,
                                  28897.138672]

        # Test with real data
        perf_real = Perfusion(self.data[::4, ::4, :, :4], self.affine,
                              moco=True)

        # Check that deformation field was calculated
        assert hasattr(perf_real, 'deformation_field')
        assert perf_real.deformation_field is not None

        pwi_stats = arraystats.ArrayStats(
            perf_real.perfusion_weighted).calculate()
        npt.assert_allclose([pwi_stats["mean"]["3D"],
                             pwi_stats["std"]["3D"],
                             pwi_stats["min"]["3D"],
                             pwi_stats["max"]["3D"]],
                            gold_standard_pwi_moco, rtol=0.1, atol=1E-3)

    def test_to_nifti(self):
        """Test the to_nifti method creates expected files."""
        perf = Perfusion(self.test_array, self.test_affine)
        
        # Create test directory or clean it if it exists
        if os.path.exists('test_output'):
            shutil.rmtree('test_output')
        os.makedirs('test_output', exist_ok=True)
        
        base_name = "perftest"
        
        # Test with specific maps
        perf.to_nifti(
            output_directory='test_output',
            base_file_name=base_name,
            maps=['mean_label', 'mean_control']
        )
        
        # Check files exist
        output_files = os.listdir('test_output')
        assert f"{base_name}_mean_label.nii.gz" in output_files
        assert f"{base_name}_mean_control.nii.gz" in output_files
        
        # Check files don't exist (not requested)
        assert f"{base_name}_perfusion_weighted.nii.gz" not in output_files
        
        # Clean test directory
        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))
        
        # Test with 'all' maps
        perf.to_nifti(
            output_directory='test_output',
            base_file_name=base_name,
            maps='all'
        )
        
        # Check all files exist
        output_files = os.listdir('test_output')
        assert f"{base_name}_mean_label.nii.gz" in output_files
        assert f"{base_name}_mean_control.nii.gz" in output_files
        assert f"{base_name}_perfusion_weighted.nii.gz" in output_files
        
        # Clean test directory
        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))
        
        # Test with empty list (no files should be created)
        perf.to_nifti(
            output_directory='test_output',
            base_file_name=base_name,
            maps=[]
        )
        assert len(os.listdir('test_output')) == 0
        
        # Test with motion correction and check deformation field
        perf_moco = Perfusion(self.data[::8, ::8, :, :2], self.affine,
                              moco=True)
        perf_moco.to_nifti(
            output_directory='test_output',
            base_file_name=base_name,
            maps=['deformation_field']
        )
        assert f"{base_name}_deformation_field.nii.gz" in os.listdir('test_output')
        
        # Delete test directory
        shutil.rmtree('test_output')

    def test_nifti_file_content(self):
        """Test that NIFTI files have correct content."""
        perf = Perfusion(self.test_array, self.test_affine)
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save perfusion weighted map
            perf.to_nifti(
                output_directory=tmpdirname,
                maps=['perfusion_weighted']
            )
            
            # Load the saved file
            saved_file_path = os.path.join(tmpdirname, "Output_perfusion_weighted.nii.gz")
            saved_nifti = nib.load(saved_file_path)
            saved_data = saved_nifti.get_fdata()
            
            # Check that the data matches
            npt.assert_array_almost_equal(
                saved_data, perf.perfusion_weighted,
                decimal=6, err_msg="Saved NIFTI data doesn't match the source data"
            )
            
            # Check that the affine matrix is preserved
            npt.assert_array_equal(
                saved_nifti.affine, perf.affine,
                err_msg="Saved NIFTI affine doesn't match the source affine"
            )

    def test_unusual_input_handling(self):
        """Test handling of unusual inputs."""
        # Test with zero data
        zero_data = np.zeros_like(self.test_array)
        perf_zero = Perfusion(zero_data, self.test_affine)
        assert np.all(perf_zero.perfusion_weighted == 0)
        
        # Test with reversed control/label (should trigger warning)
        # Create data where label > control
        reversed_data = self.test_array.copy()
        reversed_data[..., 0::2] = self.control_data[..., np.newaxis]  # Control becomes label
        reversed_data[..., 1::2] = self.label_data[..., np.newaxis]  # Label becomes control
        
        with pytest.warns(UserWarning):
            perf_reversed = Perfusion(reversed_data, self.test_affine)
            # Even with warning, calculation should proceed
            assert perf_reversed.perfusion_weighted is not None


# Delete the NIFTI test folder recursively if any of the unit tests failed
if os.path.exists('test_output'):
    shutil.rmtree('test_output')
