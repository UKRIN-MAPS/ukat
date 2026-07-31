import csv
import os
import shutil
import numpy.testing as npt
import pytest

from ukat.data import fetch
from ukat.segmentation.whole_kidney import Segmentation
from ukat.utils import arraystats


class TestSegmentation:
    image, affine = fetch.t2w_volume_philips()
    segmentation = Segmentation(image, affine)

    def test_get_mask(self):
        expected = [0.021034, 0.143496, 0.0, 1.0]
        mask = self.segmentation.get_mask()
        mask_stats = arraystats.ArrayStats(mask).calculate()
        npt.assert_allclose([mask_stats["mean"]["3D"], mask_stats["std"]["3D"],
                             mask_stats["min"]["3D"], mask_stats["max"]["3D"]],
                            expected, rtol=1e-6, atol=1e-4)

    def test_get_kidneys(self):
        expected = [0.031862, 0.229133, 0.0, 2.0]
        mask = self.segmentation.get_kidneys()
        mask_stats = arraystats.ArrayStats(mask).calculate()
        npt.assert_allclose([mask_stats["mean"]["3D"], mask_stats["std"]["3D"],
                             mask_stats["min"]["3D"], mask_stats["max"]["3D"]],
                            expected, rtol=1e-6, atol=1e-4)

    def test_get_left_kidney(self):
        expected = [0.010206, 0.100507, 0.0, 1.0]
        mask = self.segmentation.get_left_kidney()
        mask_stats = arraystats.ArrayStats(mask).calculate()
        npt.assert_allclose([mask_stats["mean"]["3D"], mask_stats["std"]["3D"],
                             mask_stats["min"]["3D"], mask_stats["max"]["3D"]],
                            expected, rtol=1e-6, atol=1e-4)

    def test_get_right_kidney(self):
        expected = [0.010828, 0.103492, 0.0, 1.0]
        mask = self.segmentation.get_right_kidney()
        mask_stats = arraystats.ArrayStats(mask).calculate()
        npt.assert_allclose([mask_stats["mean"]["3D"], mask_stats["std"]["3D"],
                             mask_stats["min"]["3D"], mask_stats["max"]["3D"]],
                            expected, rtol=1e-6, atol=1e-4)

    def test_get_volumes(self):
        expected = {'total': 221.75981201171874,
                    'left': 107.60053378582,
                    'right': 114.15927822589875}
        volumes = self.segmentation.get_volumes()
        assert volumes == expected

    def test_get_tkv(self):
        expected = 221.75981201171874
        assert self.segmentation.get_tkv() == expected

    def test_get_lkv(self):
        expected = 107.60053378582
        assert self.segmentation.get_lkv() == expected

    def test_get_rkv(self):
        expected = 114.15927822589875
        assert self.segmentation.get_rkv() == expected

    def test_save_volumes_csv(self):
        expected = [['total', 'left', 'right'],
                    ['221.75981201171874',
                     '107.60053378582',
                     '114.15927822589875']]

        if os.path.exists('test_output'):
            shutil.rmtree('test_output')
        os.makedirs('test_output', exist_ok=True)

        self.segmentation.save_volumes_csv('test_output/volumes.csv')
        output_files = os.listdir('test_output')

        assert 'volumes.csv' in output_files

        with open('test_output/volumes.csv', 'r') as csv_file:
            reader = csv.reader(csv_file)
            list_from_csv = [row for row in reader]
        assert list_from_csv == expected

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))
        shutil.rmtree('test_output')

    def test_to_nifti(self):
        if os.path.exists('test_output'):
            shutil.rmtree('test_output')
        os.makedirs('test_output', exist_ok=True)

        # Check all is saved.
        self.segmentation.to_nifti(output_directory='test_output',
                                   base_file_name='test_subject', maps='all')
        output_files = os.listdir('test_output')
        assert len(output_files) == 4
        assert 'test_subject_mask.nii.gz' in output_files
        assert 'test_subject_left_kidney.nii.gz' in output_files
        assert 'test_subject_right_kidney.nii.gz' in output_files
        assert 'test_subject_individual_kidneys.nii.gz' in output_files

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

        with pytest.raises(ValueError):
            self.segmentation.to_nifti(output_directory='test_output',
                                       base_file_name='test_subject',
                                       maps='not_an_option')

        # Delete 'test_output' folder
        shutil.rmtree('test_output')


if os.path.exists('test_output'):
    shutil.rmtree('test_output')
