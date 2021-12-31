import os
import itk
import numpy as np

from ukat.mapping.t2star import T2Star
from ukat.mapping.t1 import T1
from ukat.mapping.diffusion import ADC

def T2Star_Moco(image_array, list_arguments):
    echo_list = list_arguments[0]
    affine_array = list_arguments[1]
    mapper_loglin = T2Star(image_array, echo_list, affine_array, multithread=True, method='2p_exp')
    T2Star_Map = mapper_loglin.t2star_map
    M0_Map = mapper_loglin.m0_map
    par = [T2Star_Map, M0_Map]
    fit = []
    for te in echo_list:
        fit.append(M0_Map * np.exp(-te/T2Star_Map))
    fit = np.stack(fit, axis=-1)
    return fit, par

def T2Star_ElastixParameters(*argv):
    elastix_model_parameters = itk.ParameterObject.New()
    parameter_file = os.path.join(os.getcwd(), 'BSplines_T2star.txt')
    elastix_model_parameters.AddParameterFile(parameter_file)
    for list_parameters in argv:
        elastix_model_parameters.SetParameter(str(list_parameters[0]), str(list_parameters[1]))
    return elastix_model_parameters

def DWI_Moco(image_array, list_arguments):
    bvalues_list = list_arguments[0]
    affine_array = list_arguments[1]
    adc_mapper = ADC(image_array, affine_array, bvalues_list, ukrin_b=False)
    ADC_Map = adc_mapper.adc
    M0_Map = image_array[..., 0]
    par = [ADC_Map, M0_Map]
    fit = []
    for b_value in bvalues_list:
        fit.append(M0_Map * np.exp(-b_value * ADC_Map))
    fit = np.stack(fit, axis=-1)
    return fit , par

def T1_Moco(image_array, list_arguments):
    inversion_list = list_arguments[0]
    affine_array = list_arguments[1]
    #if 2 < len(arguments): print("true")
    tss = list_arguments[2]
    tss_axis = list_arguments[3]
    parameters = list_arguments[4]
    multithread = list_arguments[5]
    mapper = T1(image_array, inversion_list, affine_array, tss=tss, tss_axis=tss_axis, parameters=parameters, multithread=multithread)
    T1_Map = mapper.t1_map
    M0_Map = mapper.m0_map
    par = [T1_Map, M0_Map]
    fit = []
    for ti in inversion_list:
        fit.append(np.abs(M0_Map * (1 - 2 * np.exp(-ti/T1_Map))))
    fit = np.stack(fit, axis=-1)
    return fit , par
