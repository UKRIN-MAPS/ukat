import os
import shutil
import numpy as np
import numpy.testing as npt
import pytest
from ukat.data import fetch
from ukat.mapping.t1 import T1, magnitude_correct, two_param_eq, \
    two_param_abs_eq, three_param_eq, three_param_abs_eq
from ukat.utils import arraystats
from ukat.utils.tools import convert_to_pi_range


class TestT1:
    t1 = 1000
    m0 = 5000
    eff = 1.8
    t = np.linspace(200, 1000, 9)

    # The ideal signal produced by the equation M0 * (1 - 2 * exp(-t / T1))
    # where M0 = 5000 and T1 = 1000 at 9 t between 200 and 1000 ms
    correct_signal_two_param = np.array([-3187.30753078, -2408.18220682,
                                         -1703.20046036, -1065.30659713,
                                         -488.11636094, 34.14696209,
                                         506.71035883, 934.30340259,
                                         1321.20558829])
    # The ideal signal produced by the equation M0 * (1 - eff * exp(-t /
    # T1)) where M0 = 5000, eff = 1.8 and T1 = 1000 at 9 t between 200
    # and 1000 ms
    correct_signal_three_param = np.array([-2368.5767777, -1667.36398614,
                                           -1032.88041432, -458.77593741,
                                           60.69527515, 530.73226588,
                                           956.03932295, 1340.87306233,
                                           1689.08502946])
    # The ideal signal produced by the equation M0 * (1 - 2 * exp(-t / T1))
    # where M0 = 5000 and T1 = 1000 acquired over three slices at 9 t
    # between 200 and 1000 ms + a temporal slice spacing of 10 ms
    correct_signal_two_param_tss = np.array([[[-3187.30753078, -2408.18220682,
                                               -1703.20046036, -1065.30659713,
                                               -488.11636094, 34.14696209,
                                               506.71035883, 934.30340259,
                                               1321.20558829],
                                              [-3105.8424597, -2334.46956224,
                                               -1636.50250136, -1004.95578812,
                                               -433.50869074, 83.55802539,
                                               551.41933777, 974.75775966,
                                               1357.81020428],
                                              [-3025.18797962, -2261.49037074,
                                               -1570.46819815, -945.2054797,
                                               -379.44437595, 132.4774404,
                                               595.68345494, 1014.80958915,
                                               1394.05059827]],
                                             [[-3187.30753078, -2408.18220682,
                                               -1703.20046036, -1065.30659713,
                                               -488.11636094, 34.14696209,
                                               506.71035883, 934.30340259,
                                               1321.20558829],
                                              [-3105.8424597, -2334.46956224,
                                               -1636.50250136, -1004.95578812,
                                               -433.50869074, 83.55802539,
                                               551.41933777, 974.75775966,
                                               1357.81020428],
                                              [-3025.18797962, -2261.49037074,
                                               -1570.46819815, -945.2054797,
                                               -379.44437595, 132.4774404,
                                               595.68345494, 1014.80958915,
                                               1394.05059827]]
                                             ])
    # Signal with all 9 elements equal to -5000 between 200 and 1000 ms
    signal_fail_fit = -5000 * np.ones(9)
    affine = np.eye(4)

    def test_two_param_eq(self):
        # Without abs
        signal = two_param_eq(self.t, self.t1, self.m0)
        npt.assert_allclose(signal, self.correct_signal_two_param,
                            rtol=1e-6, atol=1e-8)
        # With abs
        signal = two_param_abs_eq(self.t, self.t1, self.m0)
        npt.assert_allclose(signal, np.abs(self.correct_signal_two_param),
                            rtol=1e-6, atol=1e-8)

    def test_three_param_eq(self):
        # Without abs
        signal = three_param_eq(self.t, self.t1, self.m0, self.eff)
        npt.assert_allclose(signal, self.correct_signal_three_param,
                            rtol=1e-6, atol=1e-8)
        # With abs
        signal = three_param_abs_eq(self.t, self.t1, self.m0, self.eff)
        npt.assert_allclose(signal, np.abs(self.correct_signal_three_param),
                            rtol=1e-6, atol=1e-8)

    def test_two_param_fit(self):
        # Make the signal into a 4D array
        signal_array = np.tile(self.correct_signal_two_param, (10, 10, 3, 1))

        # Multithread
        mapper = T1(signal_array, self.t, self.affine, multithread=True)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t1_map.mean() - self.t1 < 0.00001
        assert mapper.m0_map.mean() - self.m0 < 0.00001
        assert mapper.r1_map().mean() - 1 / self.t1 < 0.00001

        # Single Threaded
        mapper = T1(signal_array, self.t, self.affine, multithread=False)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t1_map.mean() - self.t1 < 0.00001
        assert mapper.m0_map.mean() - self.m0 < 0.00001
        assert mapper.r1_map().mean() - 1 / self.t1 < 0.00001

    def test_three_param_fit(self):
        # Make the signal into a 4D array
        signal_array = np.tile(self.correct_signal_three_param, (10, 10, 3, 1))

        # Multithread
        mapper = T1(signal_array, self.t, self.affine, parameters=3,
                    multithread=True)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t1_map.mean() - self.t1 < 0.00001
        assert mapper.m0_map.mean() - self.m0 < 0.00001
        assert mapper.eff_map.mean() - self.eff < 0.00005
        assert mapper.r1_map().mean() - 1 / self.t1 < 0.00001

        # Single Threaded
        mapper = T1(signal_array, self.t, self.affine, parameters=3,
                    multithread=False)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t1_map.mean() - self.t1 < 0.00001
        assert mapper.m0_map.mean() - self.m0 < 0.00001
        assert mapper.eff_map.mean() - self.eff < 0.00005
        assert mapper.r1_map().mean() - 1 / self.t1 < 0.00001

    def test_tss(self):

        mapper = T1(self.correct_signal_two_param_tss, self.t, self.affine,
                    tss=10)
        assert mapper.shape == self.correct_signal_two_param_tss.shape[:-1]
        assert mapper.t1_map.mean() - self.t1 < 0.00001
        assert mapper.m0_map.mean() - self.m0 < 0.00001
        assert mapper.r1_map().mean() - 1 / self.t1 < 0.00001

    def test_tss_axis(self):
        signal_array = np.swapaxes(self.correct_signal_two_param_tss, 0, 1)
        mapper = T1(signal_array, self.t, self.affine, tss=10, tss_axis=0)
        assert mapper.t1_map.mean() - self.t1 < 0.00001
        assert mapper.m0_map.mean() - self.m0 < 0.00001
        assert mapper.r1_map().mean() - 1 / self.t1 < 0.00001

    def test_failed_fit(self):
        # Make the signal, where the fitting is expected to fail, into 4D array
        signal_array = np.tile(self.signal_fail_fit, (10, 10, 3, 1))

        # Fail to fit using the 2 parameter equation
        mapper_two_param = T1(signal_array, self.t, self.affine,
                              parameters=2, multithread=True)
        assert mapper_two_param.shape == signal_array.shape[:-1]
        # Voxels that fail to fit are set to zero
        assert mapper_two_param.t1_map.mean() == 0.0
        assert mapper_two_param.t1_err.mean() == 0.0
        assert mapper_two_param.m0_map.mean() == 0.0
        assert mapper_two_param.m0_err.mean() == 0.0

        # Fail to fit using the 3 parameter equation
        mapper_three_param = T1(signal_array, self.t, self.affine,
                                parameters=3, multithread=True)
        assert mapper_three_param.shape == signal_array.shape[:-1]
        # Voxels that fail to fit are set to zero
        assert mapper_three_param.t1_map.mean() == 0.0
        assert mapper_three_param.t1_err.mean() == 0.0
        assert mapper_three_param.m0_map.mean() == 0.0
        assert mapper_three_param.m0_err.mean() == 0.0

    def test_mask(self):
        signal_array = np.tile(self.correct_signal_two_param, (10, 10, 3, 1))

        # Bool mask
        mask = np.ones(signal_array.shape[:-1], dtype=bool)
        mask[:5, ...] = False
        mapper = T1(signal_array, self.t, self.affine, mask=mask)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t1_map[5:, ...].mean() - self.t1 < 0.00001
        assert mapper.t1_map[:5, ...].mean() < 0.00001

        # Int mask
        mask = np.ones(signal_array.shape[:-1])
        mask[:5, ...] = 0
        mapper = T1(signal_array, self.t, self.affine, mask=mask)
        assert mapper.shape == signal_array.shape[:-1]
        assert mapper.t1_map[5:, ...].mean() - self.t1 < 0.00001
        assert mapper.t1_map[:5, ...].mean() < 0.00001

    def test_mismatched_raw_data_and_inversion_lengths(self):

        with pytest.raises(AssertionError):
            mapper = T1(pixel_array=np.zeros((5, 5, 4)),
                        inversion_list=np.linspace(0, 2000, 5),
                        affine=self.affine)

        with pytest.raises(AssertionError):
            mapper = T1(pixel_array=np.zeros((5, 5, 5)),
                        inversion_list=np.linspace(0, 2000, 4),
                        affine=self.affine)

    def test_parameters(self):

        # One parameter fit
        with pytest.raises(ValueError):
            mapper = T1(pixel_array=np.zeros((5, 5, 5)),
                        inversion_list=np.linspace(0, 2000, 5),
                        affine=self.affine, parameters=1)

        # Four parameter fit
        with pytest.raises(ValueError):
            mapper = T1(pixel_array=np.zeros((5, 5, 5)),
                        inversion_list=np.linspace(0, 2000, 5),
                        affine=self.affine, parameters=4)

    def test_tss_valid_axis(self):

        # 4D -1 index
        with pytest.raises(AssertionError):
            mapper = T1(pixel_array=np.zeros((5, 5, 5, 10)),
                        inversion_list=np.linspace(0, 2000, 10),
                        affine=self.affine, tss=1, tss_axis=-1)

        # 4D 3 index
        with pytest.raises(AssertionError):
            mapper = T1(pixel_array=np.zeros((5, 5, 5, 10)),
                        inversion_list=np.linspace(0, 2000, 10),
                        affine=self.affine, tss=1, tss_axis=3)

        # 4D 4 index
        with pytest.raises(AssertionError):
            mapper = T1(pixel_array=np.zeros((5, 5, 5, 10)),
                        inversion_list=np.linspace(0, 2000, 10),
                        affine=self.affine, tss=1, tss_axis=4)

        # 3D -1 index
        with pytest.raises(AssertionError):
            mapper = T1(pixel_array=np.zeros((5, 5, 10)),
                        inversion_list=np.linspace(0, 2000, 10),
                        affine=self.affine, tss=1, tss_axis=-1)

        # 3D 2 index
        with pytest.raises(AssertionError):
            mapper = T1(pixel_array=np.zeros((5, 5, 10)),
                        inversion_list=np.linspace(0, 2000, 10),
                        affine=self.affine, tss=1, tss_axis=2)

    def test_molli_2p_warning(self):
        with pytest.warns(UserWarning):
            mapper = T1(pixel_array=np.zeros((5, 5, 5)),
                        inversion_list=np.linspace(0, 2000, 5),
                        affine=self.affine, parameters=2, molli=True)

    def test_real_data(self):
        # Get test data
        magnitude, phase, affine, ti, tss = fetch.t1_philips(2)
        image_molli, affine_molli, ti_molli = fetch.t1_molli_philips()

        # Convert times to ms
        ti = np.array(ti) * 1000
        tss *= 1000
        ti_molli *= 1000

        # Crop to reduce runtime
        magnitude = magnitude[37:55, 65:85, :2, :]
        image_molli = image_molli[37:55, 65:85, :2, :]

        # Gold standard statistics
        gold_standard_2p = [1041.581031, 430.129308, 241.512336, 2603.911794]
        gold_standard_3p = [1416.989523, 722.097507, 0.0, 4909.693108]
        gold_standard_3p_single = [1379.242715, 714.21752, 0.0, 4308.23814]
        gold_standard_molli = [782.923767, 495.751163, 0.0, 4452.606435]

        # Two parameter method
        mapper = T1(magnitude, ti, affine, parameters=2, tss=tss)
        t1_stats = arraystats.ArrayStats(mapper.t1_map).calculate()
        npt.assert_allclose([t1_stats['mean']['3D'], t1_stats['std']['3D'],
                             t1_stats['min']['3D'], t1_stats['max']['3D']],
                            gold_standard_2p, rtol=1e-6, atol=1e-4)

        # Three parameter method
        mapper = T1(magnitude, ti, affine, parameters=3, tss=tss)
        t1_stats = arraystats.ArrayStats(mapper.t1_map).calculate()
        npt.assert_allclose([t1_stats['mean']['3D'], t1_stats['std']['3D'],
                             t1_stats['min']['3D'], t1_stats['max']['3D']],
                            gold_standard_3p, rtol=1e-4, atol=5e-2)

        # Three parameter method for first slice only
        mapper = T1(magnitude[:, :, 0, :], ti, affine, parameters=3,
                    tss=0, tss_axis=None)
        t1_stats = arraystats.ArrayStats(mapper.t1_map).calculate()
        npt.assert_allclose([t1_stats['mean'], t1_stats['std'],
                             t1_stats['min'], t1_stats['max']],
                            gold_standard_3p_single, rtol=1e-6, atol=5e-2)

        # MOLLI corrections/data
        mapper = T1(image_molli, ti_molli, affine_molli, parameters=3,
                    molli=True)
        t1_stats = arraystats.ArrayStats(mapper.t1_map).calculate()
        npt.assert_allclose([t1_stats['mean']['3D'], t1_stats['std']['3D'],
                             t1_stats['min']['3D'], t1_stats['max']['3D']],
                            gold_standard_molli, rtol=1e-6, atol=5e-3)

    def test_to_nifti(self):
        # Create a T1 map instance and test different export to NIFTI scenarios
        signal_array = np.tile(self.correct_signal_three_param, (10, 10, 3, 1))
        mapper = T1(signal_array, self.t, self.affine, parameters=3)

        os.makedirs('test_output', exist_ok=True)

        # Check all is saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='t1test', maps='all')
        output_files = os.listdir('test_output')
        assert len(output_files) == 8
        assert 't1test_eff_err.nii.gz' in output_files
        assert 't1test_eff_map.nii.gz' in output_files
        assert 't1test_m0_err.nii.gz' in output_files
        assert 't1test_m0_map.nii.gz' in output_files
        assert 't1test_mask.nii.gz' in output_files
        assert 't1test_r1_map.nii.gz' in output_files
        assert 't1test_t1_err.nii.gz' in output_files
        assert 't1test_t1_map.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        # Check that no files are saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='t1test', maps=[])
        output_files = os.listdir('test_output')
        assert len(output_files) == 0

        # Check that only t1, r1 and efficiency are saved.
        mapper.to_nifti(output_directory='test_output',
                        base_file_name='t1test', maps=['t1', 'r1', 'eff'])
        output_files = os.listdir('test_output')
        assert len(output_files) == 3
        assert 't1test_t1_map.nii.gz' in output_files
        assert 't1test_r1_map.nii.gz' in output_files
        assert 't1test_eff_map.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        # Check that it fails when no maps are given
        with pytest.raises(ValueError):
            mapper = T1(signal_array, self.t, self.affine)
            mapper.to_nifti(output_directory='test_output',
                            base_file_name='t1test', maps='')

        # Delete 'test_output' folder
        shutil.rmtree('test_output')


class TestMagnitudeCorrect:

    real = np.array([-43611, -46086, -19840, -14032, 8654])
    imag = np.array([51432, 30621, 5189, 4677, -6265])

    # Numpy complex data
    comp = real + imag * (0 + 1j)
    # "Simple data" uses the last dimension to hold real and imaginary
    # components respectively
    simple = np.array([real, imag]).T

    correct_array = np.array([-67432.70678981, -55331.41094351,
                              -20507.34797579, -14790.92130329,
                              10683.72318061])

    def test_complex_conversion_shape(self):

        # Has no effect on already complex data
        corrected = magnitude_correct(self.comp)
        assert corrected.shape == (5,)

        # Converts the last dimension to complex data
        corrected = magnitude_correct(self.simple)
        assert corrected.shape == (5,)

        # Raise error if not complex data but last dimension doesn't have
        # length two i.e. isn't real and imag
        with pytest.raises(ValueError):
            corrected = magnitude_correct(self.simple[:, 0])

    def test_input_dimensions(self):

        # Tile existing data to increase dimensions
        # Comp tested up to 4D i.e. [x, y, z, TI]
        comp_2d = np.tile(self.comp, (4, 1))
        comp_3d = np.tile(self.comp, (4, 4, 1))
        comp_4d = np.tile(self.comp, (4, 4, 4, 1))
        # Simple tested up to 5D i.e. [x, y, z, TI, re/im]
        simp_3d = np.tile(self.simple, (4, 1, 1))
        simp_4d = np.tile(self.simple, (4, 4, 1, 1))
        simp_5d = np.tile(self.simple, (4, 4, 4, 1, 1))

        corrected = magnitude_correct(comp_2d)
        assert corrected.shape == (4, 5)
        npt.assert_allclose(corrected,
                            np.tile(self.correct_array, (4, 1)),
                            rtol=1e-9, atol=1e-9)

        corrected = magnitude_correct(comp_3d)
        assert corrected.shape == (4, 4, 5)
        npt.assert_allclose(corrected,
                            np.tile(self.correct_array, (4, 4, 1)),
                            rtol=1e-9, atol=1e-9)

        corrected = magnitude_correct(comp_4d)
        assert corrected.shape == (4, 4, 4, 5)
        npt.assert_allclose(corrected,
                            np.tile(self.correct_array, (4, 4, 4, 1)),
                            rtol=1e-9, atol=1e-9)

        corrected = magnitude_correct(simp_3d)
        assert corrected.shape == (4, 5)
        npt.assert_allclose(corrected,
                            np.tile(self.correct_array, (4, 1)),
                            rtol=1e-9, atol=1e-9)

        corrected = magnitude_correct(simp_4d)
        assert corrected.shape == (4, 4, 5)
        npt.assert_allclose(corrected,
                            np.tile(self.correct_array, (4, 4, 1)),
                            rtol=1e-9, atol=1e-9)

        corrected = magnitude_correct(simp_5d)
        assert corrected.shape == (4, 4, 4, 5)
        npt.assert_allclose(corrected,
                            np.tile(self.correct_array, (4, 4, 4, 1)),
                            rtol=1e-9, atol=1e-9)

    def test_real_data(self):
        # Get test data
        magnitude, phase, affine, ti, tss = fetch.t1_philips(2)
        phase = convert_to_pi_range(phase)

        gold_standard = [59.35023, 191.800416, -2381.208749, 4347.05731]
        # Convert magnitude and phase into complex data
        complex_data = magnitude * (np.cos(phase) + 1j * np.sin(phase))

        magnitude_corrected = np.nan_to_num(magnitude_correct(complex_data))

        complex_stats = arraystats.ArrayStats(magnitude_corrected).calculate()
        npt.assert_allclose([complex_stats['mean']['4D'],
                             complex_stats['std']['4D'],
                             complex_stats['min']['4D'],
                             complex_stats['max']['4D']],
                            gold_standard, rtol=1e-6, atol=1e-4)


# Delete the NIFTI test folder recursively if any of the unit tests failed
if os.path.exists('test_output'):
    shutil.rmtree('test_output')
