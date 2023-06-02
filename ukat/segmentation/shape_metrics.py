import numpy as np
import pandas as pd

from skimage.measure import label, regionprops_table


class ShapeMetrics:
    def __init__(self, pixel_array, affine=None, zoom=None, kidneys=True):
        if (affine is None) and (zoom is None):
            raise ValueError('affine or zoom must be specified')

        self.pixel_array = pixel_array
        self.affine = affine
        self.zoom = zoom
        self.n_labels = len(np.unique(pixel_array[pixel_array > 0]))
        if self.n_labels == 1:
            self.labels = label(pixel_array)
            self.n_labels = len(np.unique(self.labels[self.labels > 0]))
        else:
            self.labels = self.pixel_array
        self.metrics = self.get_metrics()

        if kidneys:
            if self.n_labels != 2:
                # todo check this should actually be an AttributeError when you're on the ground
                raise AttributeError('Expected two labels (L and R) if kidney=True')
            self.metrics.index = ['L', 'R']

    def get_metrics(self, properties=None):
        properties_default = ['area', 'area_bbox', 'area_convex', 'area_filled', 'axis_major_length',
                              'axis_minor_length', 'euler_number', 'feret_diameter_max', 'solidity']
        if properties is not None:
            if type(properties) is str:
                properties = [properties]
            properties_default += properties
        props_df = pd.DataFrame(regionprops_table(self.labels, properties=properties_default))
        return props_df
