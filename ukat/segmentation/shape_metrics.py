import numpy as np
import pandas as pd
import trimesh

from nibabel.affines import voxel_sizes
from skimage.measure import label, marching_cubes, regionprops
from trimesh import smoothing


class ShapeMetrics:
    def __init__(self, pixel_array, affine=None, zoom=None, kidneys=True,
                 region_labels=None):
        """
        Calculate shape metrics for a mask.

        Parameters
        ----------
        pixel_array : np.ndarray(dtype=np.uint8)
            An array containing the mask, this array can either be a binary
            array of 0s and 1s where 0 represents background tissue or an
            array of integers where each integer represents a different tissue.
            If a binary mask is provided, the metrics of each isolated
            region will be calculated.
        affine : np.ndarray
            A matrix giving the relationship between voxel coordinates and
            world coordinates.
        zoom : tuple of float, shape (ndim,)
            A tuple of floats giving the voxel size in mm.
        kidneys : bool, optional
            Default True
            If true, it will be assumed that the two regions in the mask are
            the left and right kidneys.
        region_labels : list, optional
            A list of string labels corresponding to each integer label in
            pixel_array
        """
        if (affine is None) and (zoom is None):
            raise ValueError('Affine or zoom must be specified')

        self.pixel_array = pixel_array
        self.affine = affine
        if zoom is None:
            zoom = tuple(voxel_sizes(self.affine))
        self.zoom = zoom
        self.n_labels = len(np.unique(pixel_array[pixel_array > 0]))
        if self.n_labels == 1:
            self.labels = label(self.pixel_array)
            self.n_labels = len(np.unique(self.labels[self.labels > 0]))
        else:
            self.labels = self.pixel_array

        if kidneys:
            if self.n_labels != 2:
                raise ValueError('Expected two labels (L and R) if '
                                 'kidney=True')
            self.region_labels = ['L', 'R']
        elif region_labels is not None:
            if len(region_labels) != self.n_labels:
                raise ValueError('The number of labels must match the number '
                                 'of regions in the mask')
            self.region_labels = region_labels
        else:
            self.region_labels = np.arange(1, self.n_labels + 1)

        self.metrics = self.get_metrics()

    def get_metrics(self):
        """
        Calculate shape metrics for each region in the mask.

        Returns
        -------
        props_df : pd.DataFrame
            A dataframe containing the calculated shape metrics for each
            region in the mask.
        """
        properties = ['volume', 'surface_area', 'volume_bbox', 'volume_convex',
                      'volume_filled', 'n_vox', 'long_axis',
                      'short_axis', 'compactness', 'euler_number', 'solidity']
        props_df = pd.DataFrame(index=self.region_labels, columns=properties)

        for region, label in zip(self.region_labels,
                                 np.unique(self.labels[self.labels > 0])):
            props_df.loc[region] = self._get_region_props(self.labels == label)
        return props_df

    def _get_region_props(self, region):
        """
        Calculate shape metrics for a single region.

        Parameters
        ----------
        region : np.ndarray(dtype=np.bool)
            A binary array of 0s and 1s where 0 represents background tissue.

        Returns
        -------
        props_dict : dict
            A dictionary containing the calculated shape metrics for the
            region.
        """
        props = regionprops(region.astype(np.uint8), spacing=self.zoom)[0]
        mesh = self._get_smoothed_mesh(region)

        props_dict = {}
        props_dict.update({'volume': props['area'] / 1000})  # mm^3 to mL
        props_dict.update({'surface_area': mesh.area / 100})  # mm^2 to cm^2
        props_dict.update(
            {'volume_bbox': props['bbox_area'] / 1000})  # mm^3 to mL
        props_dict.update(
            {'volume_convex': props['convex_area'] / 1000})  # mm^3 to mL
        props_dict.update(
            {'volume_filled': props['filled_area'] / 1000})  # mm^3 to mL
        props_dict.update({'n_vox': props['num_pixels']})
        props_dict.update(
            {'long_axis': props['major_axis_length'] / 10})  # mm to cm
        props_dict.update(
            {'short_axis': props['minor_axis_length'] / 10})  # mm to cm
        compactness = (props_dict['volume'] / props_dict['surface_area']) / \
                      (props['equivalent_diameter_area'] / 6)
        props_dict.update({'compactness': compactness})
        props_dict.update({'euler_number': props['euler_number']})
        props_dict.update({'solidity': props['solidity']})

        return props_dict

    def _get_smoothed_mesh(self, region):
        """
        Generate a smoothed mesh from a binary region. Parameters have been
        optimised for kidneys.

        Parameters
        ----------
        region : np.ndarray(dtype=np.bool)
            A binary array of 0s and 1s where 0 represents background tissue.

        Returns
        -------
        mesh : trimesh.Trimesh
            A smoothed mesh representation of the region.
        """
        verts, faces, _, _ = marching_cubes(region.astype(np.uint8),
                                            spacing=self.zoom, level=0.5,
                                            step_size=1.0)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh = smoothing.filter_laplacian(mesh, lamb=1, iterations=20)
        return mesh
