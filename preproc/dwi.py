"""Diffusion-weighted imaging preprocessing module

"""
import SimpleITK as sitk


def get_parameter_map(disp=False, affine_dti=False):
    """Create and return elastix parameter map

    Parameters
    ----------
    disp : bool
        Print parameter map to console
    affine_dti : bool
        True:  "Transform" = "AffineDTITransform"  (i.e. affine registration)
        False: "Transform" = "EulerTransform" (i.e. rigid registration)

    Returns
    -------
    SimpleITK.SimpleITK.ParameterMap
        Parameter map

    """
    # Take the default rigid parameter map as a starting point
    parameter_map = sitk.GetDefaultParameterMap("rigid")

    # "Set the NumberOfSpatialSamples to 3000." (as per manual)
    parameter_map["NumberOfSpatialSamples"] = ["3000"]

    # Other settings
    parameter_map["MovingImageDimension"] = ["3"]
    parameter_map["FixedImageDimension"] = ["3"]
    parameter_map["ResultImageFormat"] = ["nii.gz"]
    parameter_map["ErodeMask"] = ["false"]
    parameter_map["FixedInternalImagePixelType"] = ["float"]
    parameter_map["MovingInternalImagePixelType"] = ["float"]
    parameter_map["ResultImagePixelType"] = ["float"]
    parameter_map["HowToCombineTransforms"] = ["Compose"]
    parameter_map["UseAdaptiveStepSizes"] = ["true"]
    parameter_map["UseDirectionCosines"] = ["true"]
    parameter_map["NumberOfHistogramBins"] = ["32"]

    # I have been running into the "Too many samples map outside moving image
    # buffer" problem when using a multi-resolution strategy. Our masks are
    # quite small especially in datasets with a small number of slices. Still,
    # we probably don't want to increase the size of the masks. Alternatively,
    # I experimented with reducing the number of resolutions from the default
    # value, 3, to 2, where the error still ocurred, and then 1, which ran fine
    parameter_map["NumberOfResolutions"] = ["1"]

    if affine_dti:
        parameter_map["Transform"] = ["AffineDTITransform"]

    # Possible modifications --------------------------------------------------

    # Looks like it is not possible to use Full/FullSampler by default with
    # simple elastix**.
    # **github.com/SuperElastix/SimpleElastix/issues/300#issuecomment-499497546
    # parameter_map["ImageSampler"] = ["Full"]

    # parameter_map["Interpolator"] = ["BSplineInterpolator"]
    # parameter_map["MaximumNumberOfIterations"] = ["500"]
    # parameter_map["NumberOfSpatialSamples"] = ["4096"]
    # parameter_map["BSplineInterpolationOrder"] = ["1"]

    if disp:
        sitk.PrintParameterMap(parameter_map)

    return parameter_map
