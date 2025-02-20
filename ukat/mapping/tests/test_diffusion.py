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
    pixel_array = pixel_array[35:55, 40:80, 3:5, :]
    mask = pixel_array[..., 0] > 20000

    def test_missmatched_raw_data_and_bvals(self):

        with pytest.raises(AssertionError):
            mapper = ADC(self.pixel_array, self.affine, self.bvals[:-2],
                         self.mask)

    def test_fail_to_fit(self):
        mapper = ADC(self.pixel_array[..., ::-1], self.affine, self.bvals,
                     self.mask)
        assert np.abs(mapper.adc.mean()) < 1e-6

    def test_negative_signal(self):
        gold_standard_adc = [0.001122, 0.001239, 0.0, 0.005391]
        gold_standard_adc_err = [0.000114, 0.000175, 0.0, 0.001044]
        gold_standard_adc_r2 = [0.39937, 0.432726, 0.0, 0.994381]
        negateive_pixel_array = self.pixel_array.copy()
        negateive_pixel_array[:10, :, :, :] -= 40000
        mapper = ADC(negateive_pixel_array, self.affine, self.bvals)
        adc_stats = arraystats.ArrayStats(mapper.adc).calculate()
        adc_err_stats = arraystats.ArrayStats(mapper.adc_err).calculate()
        adc_r2_stats = arraystats.ArrayStats(mapper.r2).calculate()
        assert np.sum(mapper.adc[:10, :, :]) == 0
        npt.assert_allclose([adc_stats['mean']['3D'], adc_stats['std']['3D'],
                             adc_stats['min']['3D'], adc_stats['max']['3D']],
                            gold_standard_adc, rtol=5e-4, atol=5e-7)
        npt.assert_allclose([adc_err_stats['mean']['3D'],
                             adc_err_stats['std']['3D'],
                             adc_err_stats['min']['3D'],
                             adc_err_stats['max']['3D']],
                            gold_standard_adc_err, rtol=5e-3, atol=1e-7)
        npt.assert_allclose([adc_r2_stats['mean']['3D'],
                             adc_r2_stats['std']['3D'],
                             adc_r2_stats['min']['3D'],
                             adc_r2_stats['max']['3D']],
                            gold_standard_adc_r2, rtol=5e-3, atol=1e-7)

    def test_mask_moco_error(self):
        with pytest.raises(ValueError):
            mapper = ADC(self.pixel_array, self.affine, self.bvals, self.mask,
                     moco=True)

    def test_real_data(self):
        # Gold standard statistics
        gold_standard_adc = [0.00198, 0.000855, 0.0, 0.005391]
        gold_standard_adc_err = [0.000184, 0.000165, 0.0, 0.001044]
        gold_standard_adc_moco = [0.001912, 0.000645, 0.0, 0.004614]
        gold_standard_adc_err_moco = [0.00034, 0.000319, 0.0, 0.002749]
        # Test maps, without moco
        mapper = ADC(self.pixel_array, self.affine, self.bvals, self.mask,
                     moco=False)
        adc_stats = arraystats.ArrayStats(mapper.adc).calculate()
        adc_err_stats = arraystats.ArrayStats(mapper.adc_err).calculate()
        npt.assert_allclose([adc_stats['mean']['3D'], adc_stats['std']['3D'],
                             adc_stats['min']['3D'], adc_stats['max']['3D']],
                            gold_standard_adc, rtol=5e-4, atol=5e-7)
        npt.assert_allclose([adc_err_stats['mean']['3D'],
                             adc_err_stats['std']['3D'],
                             adc_err_stats['min']['3D'],
                             adc_err_stats['max']['3D']],
                            gold_standard_adc_err, rtol=5e-3, atol=1e-7)

        # Test maps, with moco
        # Using ukrin_b=True to reduce run time.
        mapper = ADC(self.pixel_array, self.affine, self.bvals, moco=True,
                     ukrin_b=True)
        adc_stats = arraystats.ArrayStats(mapper.adc).calculate()
        adc_err_stats = arraystats.ArrayStats(mapper.adc_err).calculate()
        npt.assert_allclose([adc_stats['mean']['3D'], adc_stats['std']['3D'],
                             adc_stats['min']['3D'], adc_stats['max']['3D']],
                            gold_standard_adc_moco, rtol=5e-4, atol=5e-7)
        npt.assert_allclose([adc_err_stats['mean']['3D'],
                             adc_err_stats['std']['3D'],
                             adc_err_stats['min']['3D'],
                             adc_err_stats['max']['3D']],
                            gold_standard_adc_err_moco, rtol=5e-3, atol=1e-7)

    def test_ukrin_b(self):
        # Gold standard statistics
        gold_standard_adc = [0.001811, 0.000751, 0.0, 0.004789]
        gold_standard_adc_err = [0.000332, 0.000375, 0.0, 0.00275]
        # Test maps
        mapper = ADC(self.pixel_array, self.affine, self.bvals, self.mask,
                     ukrin_b=True)
        assert mapper.n_bvals == 4
        npt.assert_array_equal(mapper.u_bvals, np.array([0, 100, 200, 800]))
        adc_stats = arraystats.ArrayStats(mapper.adc).calculate()
        adc_err_stats = arraystats.ArrayStats(mapper.adc_err).calculate()
        npt.assert_allclose([adc_stats['mean']['3D'], adc_stats['std']['3D'],
                             adc_stats['min']['3D'], adc_stats['max']['3D']],
                            gold_standard_adc, rtol=5e-4, atol=5e-7)
        npt.assert_allclose([adc_err_stats['mean']['3D'],
                             adc_err_stats['std']['3D'],
                             adc_err_stats['min']['3D'],
                             adc_err_stats['max']['3D']],
                            gold_standard_adc_err, rtol=1e-3, atol=5e-7)

    def test_to_nifti(self):
        mapper = ADC(self.pixel_array, self.affine, self.bvals, self.mask)

        if os.path.exists('test_output'):
            shutil.rmtree('test_output')
        os.makedirs('test_output', exist_ok=True)

        # Check all is saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='adc_test', maps='all')
        output_files = os.listdir('test_output')
        assert len(output_files) == 6
        assert 'adc_test_adc_map.nii.gz' in output_files
        assert 'adc_test_adc_err.nii.gz' in output_files
        assert 'adc_test_mask.nii.gz' in output_files
        assert 'adc_test_r2.nii.gz' in output_files
        assert 'adc_test_s0_map.nii.gz' in output_files
        assert 'adc_test_s0_err.nii.gz' in output_files

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

    def test_get_fit_signal(self):
        fit_signal_gold = [42900.277636, 30374.229889,0.0, 150559.353217]
        mapper = ADC(self.pixel_array, self.affine, self.bvals, self.mask)
        fit_signal = mapper.get_fit_signal()
        stats = arraystats.ArrayStats(fit_signal).calculate()
        npt.assert_allclose([stats["mean"]["4D"], stats["std"]["4D"],
                             stats["min"]["4D"], stats["max"]["4D"]],
                            fit_signal_gold,
                            rtol=1e-6, atol=1e-4)


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

    def test_ukrin_b(self):
        # Gold standard statistics
        gold_standard_md = [0.001611, 0.001437, 0.0, 0.02951]
        gold_standard_fa = [0.446954, 0.291819, 0.0, 1.0]
        gold_standard_color_fa = [0.216434, 0.219389, 0.0, 0.99972]

        # Test maps
        mapper = DTI(self.pixel_array, self.affine, self.bvals, self.bvecs,
                     self.mask, ukrin_b=True)
        assert mapper.n_bvals == 4
        npt.assert_array_equal(mapper.u_bvals, np.array([0, 100, 200, 800]))
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

        if os.path.exists('test_output'):
            shutil.rmtree('test_output')
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
