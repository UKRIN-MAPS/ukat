"""
This module is a collection of functions where each function returns a set
of elastix registration parameters and their respective value.

These are called at the initialisation of the MotionCorrection class
in `mdr.py` and feed the registration parameters to the MDReg class
from the `mdreg` package.
"""

import itk


def DWI_BSplines(*argv):
    """
    Returns the default UKRIN registration parameters for MRI DWI sequences.

    Parameters
    ----------
    *argv : list, optional
        A list of 2 elements. The first element represents the parameter name,
        while the second element corresponds to the parameter's value. This
        ("parameter", "value") pair is added to the default parameter map.
        Eg., param_obj = DWI_BSplines(["FixedImageDimension", "3"]). This
        command returns the default DWI registration parameter, but it has
        "FixedImageDimension" set to "3" instead of "2".

    Returns
    -------
    param_obj : itk.ParameterObject
        The itk-elastix registration parameter map.
    """
    param_obj = itk.ParameterObject.New()
    parameter_map_bspline = param_obj.GetDefaultParameterMap('bspline')
    param_obj.AddParameterMap(parameter_map_bspline)
    param_obj.SetParameter("FixedInternalImagePixelType", "float")
    param_obj.SetParameter("MovingInternalImagePixelType", "float")
    param_obj.SetParameter("FixedImageDimension", "2")
    param_obj.SetParameter("MovingImageDimension", "2")
    param_obj.SetParameter("MaximumNumberOfIterations", "256")
    param_obj.SetParameter("UseDirectionCosines", "true")
    param_obj.SetParameter("Registration", "MultiResolutionRegistration")
    param_obj.SetParameter("ImageSampler", "RandomCoordinate")
    param_obj.SetParameter("Interpolator", "BSplineInterpolator")
    param_obj.SetParameter("ResampleInterpolator", "FinalBSplineInterpolator")
    param_obj.SetParameter("Resampler", "DefaultResampler")
    param_obj.SetParameter("BSplineInterpolationOrder", "1")
    param_obj.SetParameter("FinalBSplineInterpolationOrder", "1")
    param_obj.SetParameter("FixedImagePyramid", "FixedSmoothingImagePyramid")
    param_obj.SetParameter("MovingImagePyramid", "MovingSmoothingImagePyramid")
    param_obj.SetParameter("Optimizer", "AdaptiveStochasticGradientDescent")
    param_obj.SetParameter("HowToCombineTransforms", "Compose")
    param_obj.SetParameter("Transform", "BSplineTransform")
    param_obj.SetParameter("Metric", "AdvancedMeanSquares")
    param_obj.SetParameter("NumberOfHistogramBins", "32")
    param_obj.SetParameter("FinalGridSpacingInPhysicalUnits", ["50.0", "50.0"])
    param_obj.SetParameter("NumberOfResolutions", "4")
    param_obj.SetParameter("AutomaticParameterEstimation", "true")
    param_obj.SetParameter("ASGDParameterEstimationMethod", "Original")
    param_obj.SetParameter("MaximumNumberOfIterations", "500")
    param_obj.SetParameter("MaximumStepLength", "0.1")
    param_obj.SetParameter("NumberOfSpatialSamples", "2048")
    param_obj.SetParameter("NewSamplesEveryIteration", "true")
    param_obj.SetParameter("CheckNumberOfSamples", "true")
    param_obj.SetParameter("ErodeFixedMask", "false")
    param_obj.SetParameter("DefaultPixelValue", "0")
    param_obj.SetParameter("WriteResultImage", "true")
    param_obj.SetParameter("ResultImagePixelType", "float")
    param_obj.SetParameter("ResultImageFormat", "mhd")
    for list_arguments in argv:
        parameter = str(list_arguments[0])
        value = str(list_arguments[1])
        param_obj.SetParameter(parameter, value)
    return param_obj


def T1_BSplines(*argv):
    """
    Returns the default UKRIN registration parameters for MRI T1 sequences.

    Parameters
    ----------
    *argv : list, optional
        A list of 2 elements. The first element represents the parameter name,
        while the second element corresponds to the parameter's value. This
        ("parameter", "value") pair is added to the default parameter map.
        Eg., param_obj = T1_BSplines(["ErodeMask", "true"]). This
        command returns the default T1 registration parameter, but it has
        "ErodeMask" set to "true" instead of "false".

    Returns
    -------
    param_obj : itk.ParameterObject
        The itk-elastix registration parameter map.
    """
    param_obj = itk.ParameterObject.New()
    parameter_map_bspline = param_obj.GetDefaultParameterMap('bspline')
    param_obj.AddParameterMap(parameter_map_bspline)
    param_obj.SetParameter("FixedInternalImagePixelType", "float")
    param_obj.SetParameter("MovingInternalImagePixelType", "float")
    param_obj.SetParameter("FixedImageDimension", "2")
    param_obj.SetParameter("MovingImageDimension", "2")
    param_obj.SetParameter("MaximumNumberOfIterations", "256")
    param_obj.SetParameter("UseDirectionCosines", "true")
    param_obj.SetParameter("Registration", "MultiResolutionRegistration")
    param_obj.SetParameter("ImageSampler", "RandomCoordinate")
    param_obj.SetParameter("Interpolator", "BSplineInterpolator")
    param_obj.SetParameter("ResampleInterpolator", "FinalBSplineInterpolator")
    param_obj.SetParameter("Resampler", "DefaultResampler")
    param_obj.SetParameter("BSplineInterpolationOrder", "1")
    param_obj.SetParameter("FinalBSplineInterpolationOrder", "1")
    param_obj.SetParameter("FixedImagePyramid", "FixedSmoothingImagePyramid")
    param_obj.SetParameter("MovingImagePyramid", "MovingSmoothingImagePyramid")
    param_obj.SetParameter("Optimizer", "AdaptiveStochasticGradientDescent")
    param_obj.SetParameter("HowToCombineTransforms", "Compose")
    param_obj.SetParameter("Transform", "BSplineTransform")
    param_obj.SetParameter("Metric", "AdvancedMeanSquares")
    param_obj.SetParameter("NumberOfHistogramBins", "32")
    param_obj.SetParameter("FinalGridSpacingInPhysicalUnits", ["50.0", "50.0"])
    param_obj.SetParameter("NumberOfResolutions", "4")
    param_obj.SetParameter("AutomaticParameterEstimation", "true")
    param_obj.SetParameter("ASGDParameterEstimationMethod", "Original")
    param_obj.SetParameter("MaximumNumberOfIterations", "500")
    param_obj.SetParameter("MaximumStepLength", "0.1")
    param_obj.SetParameter("NumberOfSpatialSamples", "2048")
    param_obj.SetParameter("NewSamplesEveryIteration", "true")
    param_obj.SetParameter("CheckNumberOfSamples", "true")
    param_obj.SetParameter("ErodeMask", "false")
    param_obj.SetParameter("ErodeFixedMask", "false")
    param_obj.SetParameter("DefaultPixelValue", "0")
    param_obj.SetParameter("WriteResultImage", "true")
    param_obj.SetParameter("ResultImagePixelType", "float")
    param_obj.SetParameter("ResultImageFormat", "mhd")
    for list_arguments in argv:
        parameter = str(list_arguments[0])
        value = str(list_arguments[1])
        param_obj.SetParameter(parameter, value)
    return param_obj


def Custom_BSplines(dict_elastix):
    """
    Converts "dict_elastix" to an elastix registration parameter object
    with bspline transformation.

    Parameters
    ----------
    dict_elastix : dict
        A python dictionary where the key is the parameter name and the value
        corresponds to the parameter value.
        Eg., param_obj = Custom_BSplines({"ErodeMask": "true",
                                          "FixedImageDimension": "3"})
        This command returns an elastix parameter object of 2 parameters,
        with "ErodeMask" set to "true" and "FixedImageDimension" set to "3".

    Returns
    -------
    param_obj : itk.ParameterObject
        The itk-elastix registration parameter map with bspline transformation.
    """
    param_obj = itk.ParameterObject.New()
    parameter_map_bspline = param_obj.GetDefaultParameterMap('bspline')
    param_obj.AddParameterMap(parameter_map_bspline)
    for key, value in dict_elastix:
        param_obj.SetParameter(key, value)
    return param_obj


def Custom_Rigid(dict_elastix):
    """
    Converts "dict_elastix" to an elastix registration parameter object
    with rigid transformation.

    Parameters
    ----------
    dict_elastix : dict
        A python dictionary where the key is the parameter name and the value
        corresponds to the parameter value.
        Eg., param_obj = Custom_Rigid({"ErodeMask": "true",
                                          "FixedImageDimension": "3"})
        This command returns an elastix parameter object of 2 parameters,
        with "ErodeMask" set to "true" and "FixedImageDimension" set to "3".

    Returns
    -------
    param_obj : itk.ParameterObject
        The itk-elastix registration parameter map with rigid transformation.
    """
    param_obj = itk.ParameterObject.New()
    parameter_map_bspline = param_obj.GetDefaultParameterMap('rigid')
    param_obj.AddParameterMap(parameter_map_bspline)
    for key, value in dict_elastix:
        param_obj.SetParameter(key, value)
    return param_obj
