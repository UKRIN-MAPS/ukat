"""
Each class in this module corresponds to one fitting model that serves as the
core of the model-driven motion correction method.
These are called at the initialisation of the MotionCorrection class
in `mdr.py` and feed the model fitting class and arguments to the MDReg class
from the `mdreg` package.
According to the structure and rules of the `mdreg.models` module, each model
is described by a class containing 3 methods: pars(), bounds() and main().
"""

import numpy as np
from ukat.mapping.t1 import T1
from ukat.mapping.diffusion import ADC


class DWI_Moco:
    """
    Performs model fitting in DWI images when passed to an instance of MDReg().
    """
    def pars():
        """Returns a list with the labels of the DWI parameters ADC and M0."""
        return ['ADC', 'M0']

    def bounds():
        """Returns the lower and upper values to clip the parameter maps."""
        lower = [0, 0]
        upper = [1.0, np.inf]
        return lower, upper

    def main(image_array, list_arguments):
        """
        Returns the DWI signal model fit and the fitted parameters ADC and M0.
        Parameters
        ----------
        image_array : np.ndarray
            Array containing the MRI volume(s) acquired at different
            time points. Its expected shape should be 3 or 4.
        list_arguments : list
            List containing the arguments that are passed to ukat's ADC mapping
            model. Please consult the "ADC" class in the module
            `ukat.mapping.diffusion` for more information.
        Returns
        -------
        fit : np.ndarray
            The array containing the DWI signal model fitted.
        pars : list
            A list where each element is a np.ndarray containing a fitted
            parameter map. In this particular case, it returns a list with
            the images ADC and M0.
        """
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
        image_array = np.nan_to_num(image_array)
        image_array = np.reshape(image_array,
                                 (int(np.sqrt(np.shape(image_array)[0])),
                                  int(np.sqrt(np.shape(image_array)[0])),
                                  np.shape(image_array)[1]))
        adc_mapper = ADC(image_array, affine_array, bvalues_list,
                         mask=mask, ukrin_b=b_flag)
        ADC_Map = adc_mapper.adc
        M0_Map = adc_mapper.pixel_array_mean[..., 0]
        par = np.stack([ADC_Map.flatten(), M0_Map.flatten()], axis=-1)
        fit = [M0_Map * np.exp(-b_value * ADC_Map) for b_value in bvalues_list]
        fit = np.stack(fit, axis=-1)
        return fit, par


class T1_Moco:
    """
    Performs model fitting in T1 images when passed to an instance of MDReg().
    """
    def pars():
        """Returns a list with the labels of the T1 parameters T1Map and M0."""
        return ['T1Map', 'M0']

    def bounds():
        """Returns the lower and upper values to clip the parameter maps."""
        lower = [0, 0]
        upper = [3000, np.inf]
        return lower, upper

    def main(image_array, list_arguments):
        """
        Returns the T1 signal model fit and the fitted parameters T1Map and M0.
        Parameters
        ----------
        image_array : np.ndarray
            Array containing the MRI volume(s) acquired at different
            time points. Its expected shape should be 3 or 4.
        list_arguments : list
            List containing the arguments that are passed to ukat's T1 mapping
            model. Please consult the "T1" class in the module
            `ukat.mapping.t1` for more information.
        Returns
        -------
        fit : np.ndarray
            The array containing the T1 signal model fitted.
        pars : list
            A list where each element is a np.ndarray containing a fitted
            parameter map. In this particular case, it returns a list with
            the images T1Map and M0.
        """
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
        image_array = np.nan_to_num(image_array)
        image_array = np.reshape(image_array,
                                 (int(np.sqrt(np.shape(image_array)[0])),
                                  int(np.sqrt(np.shape(image_array)[0])),
                                  np.shape(image_array)[1]))
        t1_mapper = T1(image_array, inversion_list, affine_array, tss=tss,
                       tss_axis=tss_axis, mask=mask, parameters=parameters,
                       multithread=multithread)
        T1_Map = t1_mapper.t1_map
        M0_Map = t1_mapper.m0_map
        if parameters == 3:
            Eff_Map = t1_mapper.eff_map
        par = np.stack([T1_Map.flatten(), M0_Map.flatten()], axis=-1)
        fit = np.zeros(np.shape(image_array))
        for index, _ in np.ndenumerate(image_array[..., 0]):
            m0 = M0_Map[index]
            t1 = T1_Map[index]
            if parameters == 3:
                eff = Eff_Map[index]
            signal = image_array
            for i in index:
                signal = signal[i]
            min_value = np.nanmin(signal)
            for idx, ti in enumerate(inversion_list):
                i = list(index) + [idx]
                if min_value >= 0 and parameters == 2:
                    fit[tuple(i)] = np.abs(m0 * (1 - 2 * np.exp(-ti/t1)))
                elif min_value < 0 and parameters == 2:
                    fit[tuple(i)] = m0 * (1 - 2 * np.exp(-ti/t1))
                elif min_value >= 0 and parameters == 3:
                    fit[tuple(i)] = np.abs(m0 * (1 - eff * np.exp(-ti/t1)))
                elif min_value < 0 and parameters == 3:
                    fit[tuple(i)] = m0 * (1 - eff * np.exp(-ti/t1))
        return fit, par
