"""
Description/Comment about this file

Very short description for DWI_Splines + T1.

Longer description for Custom_BSplines
"""
import itk


def DWI_BSplines(*argv):
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

def Custom_BSplines(dictionary):
    param_obj = itk.ParameterObject.New()
    parameter_map_bspline = param_obj.GetDefaultParameterMap('bspline')
    param_obj.AddParameterMap(parameter_map_bspline)
    for key, value in dictionary:
        param_obj.SetParameter(key, value)
    return param_obj