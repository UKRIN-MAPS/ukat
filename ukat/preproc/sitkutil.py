"""Tools relying on Simple ITK

The tools in this module should be general enough to be useful in the
preprocessing of multiple types of MR data (e.g. diffusion, ASL, etc...)

"""
import SimpleITK as sitk


def get3D_from4D(img, idx):
    """Extract a 3D image from the 4th dimension of a 4D image

    Parameters
    ----------
    img : SimpleITK.SimpleITK.Image
        4D SimpleITK image.
    idx : int
        index of the 3D image (volume) to extract

    Returns
    -------
    SimpleITK.SimpleITK.Image
        3D SimpleITK image: extracted volume

    Notes
    -----
    Uses SimpleITK's ExtractImageFilter [1]_. Other files are in [2]_.

    References
    ----------
    .. [1] https://itk.org/Doxygen/html/classitk_1_1ExtractImageFilter.html
    .. [2] https://simpleitk.readthedocs.io/en/next/Documentation/docs/source/filters.html

    """
    Extractor = sitk.ExtractImageFilter()
    Extractor.SetSize(img.GetSize()[:3] + (0,))
    Extractor.SetIndex([0, 0, 0, idx])
    img_extracted = Extractor.Execute(img)

    return img_extracted
