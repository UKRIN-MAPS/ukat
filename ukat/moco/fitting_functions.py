"""
Description about what this file is and why it's necessary.

Describe the input arguments for each individual function.
"""

import numpy as np
from ukat.mapping.t1 import T1
from ukat.mapping.diffusion import ADC


def DWI_Moco(image_array, list_arguments):
    affine_array = list_arguments[0]
    bvalues_list = list_arguments[1]
    if len(list_arguments) > 2:
        mask = list_arguments[2]
    else:
        mask = None
    if len(list_arguments) > 3:
        b_flag = list_arguments[3]
    else:
        b_flag = False
    image_array = np.reshape(image_array, (int(np.sqrt(np.shape(image_array)[0])), int(np.sqrt(np.shape(image_array)[0])), np.shape(image_array)[1]))
    print(np.shape(mask))
    print(np.shape(image_array))
    adc_mapper = ADC(image_array, affine_array, bvalues_list,
                     mask=mask, ukrin_b=b_flag)
    ADC_Map = adc_mapper.adc
    M0_Map = image_array[..., 0]
    par = [ADC_Map, M0_Map]
    fit = [M0_Map * np.exp(-b_value * ADC_Map) for b_value in bvalues_list]
    fit = np.stack(fit, axis=-1)
    return fit, par


def T1_Moco(image_array, list_arguments):
    affine_array = list_arguments[0]
    inversion_list = list_arguments[1]
    if len(list_arguments) > 2:
        tss = list_arguments[2]
    else:
        tss = 0
    if len(list_arguments) > 3:
        tss_axis = list_arguments[3]
    else:
        tss_axis = -2
    if len(list_arguments) > 4:
        mask = list_arguments[4]
    else:
        mask = None
    if len(list_arguments) > 5:
        parameters = list_arguments[5]
    else:
        parameters = 2
    if len(list_arguments) > 6:
        multithread = list_arguments[6]
    else:
        multithread = True
    image_array = np.reshape(image_array, (int(np.sqrt(np.shape(image_array)[0])), int(np.sqrt(np.shape(image_array)[0])), np.shape(image_array)[1]))
    t1_mapper = T1(image_array, inversion_list, affine_array, tss=tss,
                   tss_axis=tss_axis, mask=mask, parameters=parameters,
                   multithread=multithread)
    T1_Map = t1_mapper.t1_map
    M0_Map = t1_mapper.m0_map
    if parameters == 3:
        Eff_Map = t1_mapper.eff_map
    par = [T1_Map, M0_Map]
    min_value = np.nanmin(image_array)
    if min_value > 0 and parameters == 2:
        fit = [np.abs(M0_Map * (1 - 2 * np.exp(-ti/T1_Map)))
               for ti in inversion_list]
    elif min_value <= 0 and parameters == 2:
        fit = [M0_Map * (1 - 2 * np.exp(-ti/T1_Map)) for ti in inversion_list]
    elif min_value > 0 and parameters == 3:
        fit = [np.abs(M0_Map * (1 - Eff_Map * np.exp(-ti/T1_Map)))
               for ti in inversion_list]
    elif min_value <= 0 and parameters == 3:
        fit = [M0_Map * (1 - Eff_Map * np.exp(-ti/T1_Map))
               for ti in inversion_list]
    fit = np.stack(fit, axis=-1)
    return fit, par
