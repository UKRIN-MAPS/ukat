import os
import shutil
from ukat.mapping.diffusion import make_gradient_scheme, ADC, DTI
from ukat.data import fetch
from ukat.utils import arraystats
import numpy.testing as npt
import numpy as np
import pytest


class TestMakeGradientScheme:

    def test_one_bzero_true_with_bzero(self):
        bvals = [0, 5, 10]
        bvecs = [[1, 0, 1],
                 [-1, 1, 0]]
        output = make_gradient_scheme(bvals, bvecs, one_bzero=True)
        expected = (" 0.70710678          0.0   0.70710678      0\n"
                    " 0.70710678          0.0   0.70710678      5\n"
                    " 0.70710678          0.0   0.70710678     10\n"
                    "-0.70710678   0.70710678          0.0      5\n"
                    "-0.70710678   0.70710678          0.0     10")
        assert output == expected

    def test_one_bzero_true_without_bzero(self):
        bvals = [5, 10]
        bvecs = [[1, 0, 1],
                 [-1, 1, 0]]
        output = make_gradient_scheme(bvals, bvecs, one_bzero=True)
        expected = (" 0.70710678          0.0   0.70710678      0\n"
                    " 0.70710678          0.0   0.70710678      5\n"
                    " 0.70710678          0.0   0.70710678     10\n"
                    "-0.70710678   0.70710678          0.0      5\n"
                    "-0.70710678   0.70710678          0.0     10")
        assert output == expected

    def test_one_bzero_true_with_bzero_dont_normalize(self):
        bvals = [0, 5, 10]
        bvecs = [[1, 0, 1],
                 [-1, 1, 0]]
        output = make_gradient_scheme(bvals, bvecs, normalize=False,
                                      one_bzero=True)
        expected = ("          1            0            1      0\n"
                    "          1            0            1      5\n"
                    "          1            0            1     10\n"
                    "         -1            1            0      5\n"
                    "         -1            1            0     10")
        assert output == expected

    def test_one_bzero_false_with_bzero(self):
        bvals = [0, 5, 10]
        bvecs = [[1, 0, 1],
                 [-1, 1, 0]]
        output = make_gradient_scheme(bvals, bvecs, one_bzero=False)
        expected = (" 0.70710678          0.0   0.70710678      0\n"
                    " 0.70710678          0.0   0.70710678      5\n"
                    " 0.70710678          0.0   0.70710678     10\n"
                    "-0.70710678   0.70710678          0.0      0\n"
                    "-0.70710678   0.70710678          0.0      5\n"
                    "-0.70710678   0.70710678          0.0     10")
        assert output == expected

    def test_one_bzero_false_without_bzero(self):
        bvals = [5, 10]
        bvecs = [[1, 0, 1],
                 [-1, 1, 0]]
        output = make_gradient_scheme(bvals, bvecs, one_bzero=False)
        expected = (" 0.70710678          0.0   0.70710678      5\n"
                    " 0.70710678          0.0   0.70710678     10\n"
                    "-0.70710678   0.70710678          0.0      5\n"
                    "-0.70710678   0.70710678          0.0     10")
        assert output == expected


class TestADC:
    pixel_array, affine, bvals, bvecs = fetch.dwi_philips()
    pixel_array = pixel_array[35:95, 40:90, 3:6, :]
    mask = pixel_array[..., 0] > 20000

    def test_missmatched_raw_data_and_bvals(self):

        with pytest.raises(AssertionError):
            mapper = ADC(self.pixel_array, self.affine, self.bvals[:-2],
                         self.bvecs, self.mask)

    def test_missmatched_raw_data_and_bvecs(self):

        with pytest.raises(AssertionError):
            mapper = ADC(self.pixel_array, self.affine, self.bvals,
                         self.bvecs[:-1, :], self.mask)

    def test_bvecs_transpose(self):

        with pytest.warns(UserWarning):
            mapper = ADC(self.pixel_array, self.affine, self.bvals,
                         self.bvecs.T, self.mask)

    def test_fail_to_fit(self):
        mapper = ADC(self.pixel_array[..., ::-1], self.affine, self.bvals,
                     self.bvecs.T, self.mask)
        assert np.abs(mapper.adc.mean()) < 1e-6

    def test_real_data(self):
        # Gold standard statistics
        gold_standard_adc = [0.00146, 0.001057, 0.0, 0.005391]
        gold_standard_adc_err = [0.000128, 0.000143, 0.0, 0.001044]
        # Test maps
        mapper = ADC(self.pixel_array, self.affine, self.bvals, self.bvecs,
                     self.mask)
        adc_stats = arraystats.ArrayStats(mapper.adc).calculate()
        adc_err_stats = arraystats.ArrayStats(mapper.adc_err).calculate()
        npt.assert_allclose([adc_stats['mean']['3D'], adc_stats['std']['3D'],
                             adc_stats['min']['3D'], adc_stats['max']['3D']],
                            gold_standard_adc, rtol=1e-4, atol=1e-7)
        npt.assert_allclose([adc_err_stats['mean']['3D'],
                             adc_err_stats['std']['3D'],
                             adc_err_stats['min']['3D'],
                             adc_err_stats['max']['3D']],
                            gold_standard_adc_err, rtol=5e-3, atol=1e-7)

    def test_no_bvecs(self):
        # Gold standard statistics
        gold_standard_adc = [0.00146, 0.001057, 0.0, 0.005391]
        gold_standard_adc_err = [0.000128, 0.000143, 0.0, 0.001044]
        # Calculate mean across bvecs before mapping to simulate scanner
        # averaging of ADC data
        u_bvals = np.unique(self.bvals)
        pixel_array_mean = np.zeros((*self.pixel_array.shape[:3],
                                     len(u_bvals)))
        for ind, bval in enumerate(u_bvals):
            pixel_array_mean[..., ind] \
                = np.mean(self.pixel_array[..., self.bvals == bval], axis=-1)

        # Calculate map
        mapper = ADC(pixel_array_mean, self.affine, u_bvals, bvecs=None,
                     mask=self.mask)
        adc_stats = arraystats.ArrayStats(mapper.adc).calculate()
        adc_err_stats = arraystats.ArrayStats(mapper.adc_err).calculate()
        npt.assert_allclose([adc_stats['mean']['3D'], adc_stats['std']['3D'],
                             adc_stats['min']['3D'], adc_stats['max']['3D']],
                            gold_standard_adc, rtol=1e-4, atol=1e-7)
        npt.assert_allclose([adc_err_stats['mean']['3D'],
                             adc_err_stats['std']['3D'],
                             adc_err_stats['min']['3D'],
                             adc_err_stats['max']['3D']],
                            gold_standard_adc_err, rtol=5e-3, atol=1e-7)

    def test_to_nifti(self):
        mapper = ADC(self.pixel_array, self.affine, self.bvals, self.bvecs,
                     self.mask)

        os.makedirs('test_output', exist_ok=True)

        # Check all is saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='adc_test', maps='all')
        output_files = os.listdir('test_output')
        assert len(output_files) == 3
        assert 'adc_test_adc_map.nii.gz' in output_files
        assert 'adc_test_adc_err.nii.gz' in output_files
        assert 'adc_test_mask.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        # Check that no files are saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='adc_test', maps=[])
        output_files = os.listdir('test_output')
        assert len(output_files) == 0

        # Check that only adc and adc_err are saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='adc_test', maps=['adc', 'adc_err'])
        output_files = os.listdir('test_output')
        assert len(output_files) == 2
        assert 'adc_test_adc_map.nii.gz' in output_files
        assert 'adc_test_adc_err.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        # Check that it fails when no maps are given
        with pytest.raises(ValueError):
            mapper.to_nifti(output_directory='test_output',
                            base_file_name='adc_test', maps='')

        # Delete 'test_output' folder
        shutil.rmtree('test_output')


class TestDTI:
    pixel_array, affine, bvals, bvecs = fetch.dwi_philips()
    pixel_array = pixel_array[35:95, 40:90, 3:6, :]
    mask = pixel_array[..., 0] > 20000

    def test_missmatched_raw_data_and_bvals(self):

        with pytest.raises(AssertionError):
            mapper = DTI(self.pixel_array, self.affine, self.bvals[:-2],
                         self.bvecs, self.mask)

    def test_missmatched_raw_data_and_bvecs(self):

        with pytest.raises(AssertionError):
            mapper = DTI(self.pixel_array, self.affine, self.bvals,
                         self.bvecs[:-1, :], self.mask)

    def test_bvecs_transpose(self):

        with pytest.warns(UserWarning):
            mapper = DTI(self.pixel_array, self.affine, self.bvals,
                         self.bvecs.T, self.mask)

    def test_real_data(self):
        # Gold standard statistics
        gold_standard_md = [0.001781, 0.001567, 0.0, 0.012655]
        gold_standard_fa = [0.353293, 0.256178, 0.0, 0.999999]
        gold_standard_color_fa = [0.170594, 0.185415, 0.0, 0.968977]

        # Test maps
        mapper = DTI(self.pixel_array, self.affine, self.bvals, self.bvecs,
                     self.mask)
        md_stats = arraystats.ArrayStats(mapper.md).calculate()
        fa_stats = arraystats.ArrayStats(mapper.fa).calculate()
        color_fa_stats = arraystats.ArrayStats(mapper.color_fa).calculate()
        npt.assert_allclose([md_stats['mean']['3D'], md_stats['std']['3D'],
                             md_stats['min']['3D'], md_stats['max']['3D']],
                            gold_standard_md, rtol=1e-6, atol=1e-4)
        npt.assert_allclose([fa_stats['mean']['3D'], fa_stats['std']['3D'],
                             fa_stats['min']['3D'], fa_stats['max']['3D']],
                            gold_standard_fa, rtol=1e-6, atol=1e-4)
        npt.assert_allclose([color_fa_stats['mean']['4D'],
                             color_fa_stats['std']['4D'],
                             color_fa_stats['min']['4D'],
                             color_fa_stats['max']['4D']],
                            gold_standard_color_fa, rtol=1e-6, atol=1e-4)

    def test_to_nifti(self):
        mapper = DTI(self.pixel_array, self.affine, self.bvals, self.bvecs,
                     self.mask)

        os.makedirs('test_output', exist_ok=True)

        # Check all is saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='dti_test', maps='all')
        output_files = os.listdir('test_output')
        assert len(output_files) == 4
        assert 'dti_test_md_map.nii.gz' in output_files
        assert 'dti_test_fa_map.nii.gz' in output_files
        assert 'dti_test_color_fa_map.nii.gz' in output_files
        assert 'dti_test_mask.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        # Check that no files are saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='dti_test', maps=[])
        output_files = os.listdir('test_output')
        assert len(output_files) == 0

        # Check that only md and fa are saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='dti_test', maps=['md', 'fa'])
        output_files = os.listdir('test_output')
        assert len(output_files) == 2
        assert 'dti_test_md_map.nii.gz' in output_files
        assert 'dti_test_fa_map.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        # Check that it fails when no maps are given
        with pytest.raises(ValueError):
            mapper.to_nifti(output_directory='test_output',
                            base_file_name='dti_test', maps='')

        # Delete 'test_output' folder
        shutil.rmtree('test_output')


# Delete the NIFTI test folder recursively if any of the unit tests failed
if os.path.exists('test_output'):
    shutil.rmtree('test_output')
