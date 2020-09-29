"""This module implements the RegionOfInterest class which provides utilities
to work with binary masks (regions of interest) stored as NIfTI files.

"""
import numpy as np
import nibabel as nib

# Allowed values in ROIs
ROI_VALUES = np.array([0.0, 1.0])


class RegionOfInterest():
    """Region of interest utilities with NIfTI I/O

    Parameters
    ----------
    filepaths : list
        List of file paths corresponding to the NIfTI files containing the ROIs

    Attributes
    ----------
    data : list
        list of ROIs (numpy arrays, dtype bool)
    affine : (4, 4) numpy array
        affine matrix (common to all ROIs)

    Notes
    --------
    The idea for this class is to group useful methods to operate with ROIs and
    allow them to be called sequentially to build simple pipelines. Examples
    of potential pipelines could be as follows (note some of the methods aren't
    implemented yet and are just placeholders for now):

    Example 1: Create a mask for both kidneys from two separate NIfTI files
    containing respectively left and right kidney masks:
    >> ROI = RegionOfInterest(right_kidney_mask_path, left_kidney_mask_path)
    >> ROI.sum()
    >> ROI.save(both_kidneys_mask_path) # optional

    # Example 2: Create a single bounding box mask containing the two kidneys
    # from two separate NIfTI files containing respectively left and right
    # kidney masks:
    >> ROI = RegionOfInterest(nii_rkmask, nii_lkmask)
    >> ROI.sum()
    >> ROI.boundingbox()
    >> ROI.dilate() # optional
    >> ROI.threshold() # optional
    >> ROI.save(boundingbox_mask_path) # optional

    """

    def __init__(self, filepaths):
        """Constructor for RegionOfInterest class, see class docstring

        """
        # Load data
        data = []
        affine = []
        for filepath in filepaths:
            img = nib.load(filepath)
            data.append(img.get_fdata())
            affine.append(img.affine)

        # Check data consistency
        shapes = [arr.shape for arr in data]
        if not all(x == shapes[0] for x in shapes):
            raise ValueError('Images in `filepaths` must have same shape')

        if not all((x == affine[0]).all() for x in affine):
            raise ValueError('Images in `filepaths` must have same affines')

        # Ensure input images are actually ROIs (only contain 0's and 1's)
        uniques = [np.unique(x) for x in data]
        only_0s_and_1s = all([(y == ROI_VALUES).all() for y in uniques])
        if not only_0s_and_1s:
            raise ValueError('Array elements of images in `filepaths` '
                             'must be only zeros and ones')

        # Make each array in data a bool
        data = [x.astype('bool') for x in data]

        self.data = data

        # From here onwards assume that no matter how many data volumes we have
        # we just have to store a single affine, as all of them are equal
        self.affine = affine[0]

    def add(self):
        """Adds ROIsChecks ROIs do not intersect each other and adds them

        Notes
        --------
        1. Before adding the ROIs, this method checks they do not intersect
           each other
        2. A use case for this is to take 2 NIfTI files (with left/right kidney
           ROIs) and add them as sometimes it is helpful to have a "both"
           kidneys ROI for analysis/visualisation/...

        """
        data = sum(self.data)

        if np.max(data) > 1:
            raise ValueError('ROIs can not intersect each other')

        self.data = [data.astype('bool')]

    def boundingbox(self):
        """to be implemented"""

    def dilate(self):
        """to be implemented"""

    def threshold(self):
        """to be implemented"""

    def save(self, filepaths):
        """Save ROIs as NIfTI files

        Parameters
        ----------
        filepaths : list
            List of file paths to generate the NIfTI files containing the ROIs

        Notes
        --------
        This is used to save modified ROIs (e.g. after adding, thresholding...)

        """
        # Length of filepaths must be the same as data
        if len(filepaths) != len(self.data):
            raise ValueError('`filepaths` must contain a number of paths '
                             'equal to the amount of ROIs in `data`')

        for i, filepath in enumerate(filepaths):
            nib.save(nib.Nifti1Image(self.data[i], self.affine), filepath)
