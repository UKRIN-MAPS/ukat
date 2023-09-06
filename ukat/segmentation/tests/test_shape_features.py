import csv
import os
import shutil

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from ukat.data import fetch
from ukat.segmentation.shape_features import ShapeFeatures
from ukat.segmentation.whole_kidney import Segmentation
from ukat.utils import arraystats


class TestShapeFeatures:
    image, affine = fetch.t2w_volume_philips()
    zoom = (1.5000001667213594, 1.5000000306853905, 5.49999480801915)
    segmentation = Segmentation(image, affine)
    mask = segmentation.get_mask()
    kidneys = segmentation.get_kidneys()

    mask_two_body = np.zeros((50, 50, 10))
    mask_two_body[5:15, 5:45, :] = 1
    mask_two_body[20:30, 5:45, :] = 1

    mask_three_body = np.zeros((50, 50, 10))
    mask_three_body[5:15, 5:45, :] = 1
    mask_three_body[20:30, 5:45, :] = 1
    mask_three_body[35:45, 5:45, :] = 1

    simulated_affine = np.eye(4)

    # def test_get_smoothed_mesh(self):
    #     expected = [175.680058, 104.798285, 3.267293, 291.491921]
    #     shape_features = ShapeFeatures(self.kidneys, self.affine)
    #     mesh = shape_features._get_smoothed_mesh(self.kidneys == 1)
    #
    #     assert mesh.is_watertight
    #     vertex_stats = arraystats.ArrayStats(mesh.vertices).calculate()
    #     npt.assert_allclose([vertex_stats["mean"], vertex_stats["std"],
    #                          vertex_stats["min"], vertex_stats["max"]],
    #                         expected, rtol=1e-6, atol=1e-4)
    #
    # def test_get_region_props(self):
    #     shape_features = ShapeFeatures(self.kidneys, self.affine)
    #     props_dict = shape_features._get_region_props(self.kidneys == 1)
    #     assert props_dict == pytest.approx({'volume': 118.19352898042803,
    #                                         'surface_area': 148.05689835989392,
    #                                         'volume_bbox': 360.8547068442343,
    #                                         'volume_convex': 170.52736146479933,
    #                                         'volume_filled': 118.19352898042803,
    #                                         'n_vox': 9551,
    #                                         'long_axis': 11.793750181329315,
    #                                         'short_axis': 4.347012606736469,
    #                                         'compactness': 0.07866555492167773,
    #                                         'euler_number': 2,
    #                                         'solidity': 0.6931059506531204},
    #                                        rel=1e-20, abs=1e-4)
    #
    # def test_shape_features_labels_affine(self):
    #     shape_features = ShapeFeatures(self.kidneys, self.affine)
    #     features_df = shape_features.get_features()
    #     gold_df = pd.DataFrame(index=['L', 'R'],
    #                            columns=['volume', 'surface_area',
    #                                     'volume_bbox', 'volume_convex',
    #                                     'volume_filled', 'n_vox',
    #                                     'long_axis', 'short_axis',
    #                                     'compactness', 'euler_number',
    #                                     'solidity'],
    #                            data=[[118.19352898042803, 148.05689835989392,
    #                                   360.8547068442343,
    #                                   170.52736146479933, 118.19352898042803,
    #                                   9551.0,
    #                                   11.793750181329315, 4.347012606736469,
    #                                   0.07866555492167773, 2.0,
    #                                   0.6931059506531204],
    #                                  [121.80702604484902, 154.5164344861936,
    #                                   263.7357857429465,
    #                                   150.36850284171092, 121.80702604484902,
    #                                   9843.0,
    #                                   12.317681104489647, 3.6430077535636998,
    #                                   0.0769055483268716, 2.0,
    #                                   0.8100567854497571]])
    #     pd.testing.assert_frame_equal(features_df, gold_df, check_dtype=False)
    #
    # def test_shape_features_labels_zoom(self):
    #     shape_features = ShapeFeatures(self.kidneys, zoom=self.zoom)
    #     features_df = shape_features.get_features()
    #     gold_df = pd.DataFrame(index=['L', 'R'],
    #                            columns=['volume', 'surface_area',
    #                                     'volume_bbox', 'volume_convex',
    #                                     'volume_filled', 'n_vox',
    #                                     'long_axis', 'short_axis',
    #                                     'compactness', 'euler_number',
    #                                     'solidity'],
    #                            data=[[118.19352898042803, 148.05689835989392,
    #                                   360.8547068442343,
    #                                   170.52736146479933, 118.19352898042803,
    #                                   9551.0,
    #                                   11.793750181329315, 4.347012606736469,
    #                                   0.07866555492167773, 2.0,
    #                                   0.6931059506531204],
    #                                  [121.80702604484902, 154.5164344861936,
    #                                   263.7357857429465,
    #                                   150.36850284171092, 121.80702604484902,
    #                                   9843.0,
    #                                   12.317681104489647, 3.6430077535636998,
    #                                   0.0769055483268716, 2.0,
    #                                   0.8100567854497571]])
    #     pd.testing.assert_frame_equal(features_df, gold_df, check_dtype=False)
    #
    # def test_shape_features_mask(self):
    #     shape_features = ShapeFeatures(self.mask_two_body,
    #                                    self.simulated_affine)
    #     features_df = shape_features.get_features()
    #     gold_df = pd.DataFrame(index=['L', 'R'],
    #                            columns=['volume', 'surface_area',
    #                                     'volume_bbox', 'volume_convex',
    #                                     'volume_filled', 'n_vox',
    #                                     'long_axis', 'short_axis',
    #                                     'compactness', 'euler_number',
    #                                     'solidity'],
    #                            data=[[4.0, 6.354534260930347, 4.0, 4.0, 4.0,
    #                                   4000.0, 5.162363799656123,
    #                                   1.284523257866513, 0.1917669347647048,
    #                                   1.0, 1.0],
    #                                  [4.0, 6.413147961900247, 4.0, 4.0, 4.0,
    #                                   4000.0, 5.162363799656123,
    #                                   1.284523257866513, 0.1900142588811934,
    #                                   1.0, 1.0]])
    #     pd.testing.assert_frame_equal(features_df, gold_df, check_dtype=False)

    def test_no_zoom_no_affine(self):
        with pytest.raises(ValueError):
            ShapeFeatures(self.kidneys)

    # def test_region_labels_length_doesnt_match(self):
    #
    #     with pytest.raises(ValueError):
    #         ShapeFeatures(self.mask_three_body, self.simulated_affine,
    #                       region_labels=['L', 'R'],
    #                       kidneys=False)
    #
    # def test_custom_labels(self):
    #     shape_features = ShapeFeatures(self.mask, self.affine,
    #                                    region_labels=['L', 'R', 'O'],
    #                                    kidneys=False)
    #     assert np.all(shape_features.get_features().index == ['L', 'R', 'O'])
    #
    # def test_no_labels_not_kidneys(self):
    #     shape_features = ShapeFeatures(self.mask_three_body,
    #                                    self.simulated_affine,
    #                                    kidneys=False)
    #     assert np.all(shape_features.get_features().index == [1, 2, 3])
    #
    # def test_three_regions_kidney(self):
    #     with pytest.raises(ValueError):
    #         ShapeFeatures(self.mask_three_body, self.simulated_affine,
    #                       kidneys=True)

    # def test_save_csv(self):
    #     shape_features = ShapeFeatures(self.kidneys, self.affine)
    #     expected_cols = ['', 'volume', 'surface_area', 'volume_bbox',
    #                      'volume_convex', 'volume_filled', 'n_vox',
    #                      'long_axis', 'short_axis', 'compactness',
    #                      'euler_number', 'solidity']
    #
    #     if os.path.exists('test_output'):
    #         shutil.rmtree('test_output')
    #     os.makedirs('test_output', exist_ok=True)
    #
    #     shape_features.save_features_csv('test_output/features.csv')
    #     output_files = os.listdir('test_output')
    #
    #     assert 'features.csv' in output_files
    #
    #     with open('test_output/features.csv', 'r') as csv_file:
    #         reader = csv.reader(csv_file)
    #         list_from_csv = [row for row in reader]
    #     assert list_from_csv[0] == expected_cols
    #     assert len(list_from_csv) == 3
    #     assert len(list_from_csv[1]) == 12
    #     assert list_from_csv[1][0] == 'L'
    #     assert list_from_csv[2][0] == 'R'
    #
    #     for f in os.listdir('test_output'):
    #         os.remove(os.path.join('test_output', f))
    #     shutil.rmtree('test_output')
