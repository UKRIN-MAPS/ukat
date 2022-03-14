"""
The MotionCorrection class uses the MDR_Library Python Package
(https://github.com/QIB-Sheffield/MDR_Library),
which is based on the scientific work:

"Model-driven registration outperforms groupwise model-free registration for
motion correction of quantitative renal MRI"

Fotios Tagkalakis, Kanishka Sharma, Steven Sourbron, Sven Plein
https://etheses.whiterose.ac.uk/28869/
"""


import os
import numpy as np
import nibabel as nib
from MDR.MDR import model_driven_registration
import ukat.moco.fitting_functions as fitting_functions
from ukat.moco.elastix_parameters import (DWI_BSplines, T1_BSplines,
                                          Custom_BSplines)


class MotionCorrection:
    """
    Attributes
    ----------
    volumes : dict
        Total, left and right kidney volumes in milliliters.
    """

    def __init__(self, pixel_array, affine, fitting_function_name, model_input,
                 mask=None, convergence=1, elastix_params=None,
                 multithread=False, log=False):
        """Initialise a whole kidney segmentation class instance.
        Parameters
        ----------
        pixel_array : np.ndarray
            Array containing a T2-weighted FSE image.
        affine : np.ndarray
            A matrix giving the relationship between voxel coordinates and
            world coordinates.
        post_process : bool, optional
            Default True
            Keep only the two largest connected volumes in the mask. Note
            this may cause issue with subjects that have more or less than
            two kidneys.
        """
        self._mdr_results = []
        self.shape = pixel_array.shape
        if mask is None:
            self.mask = np.ones(self.shape, dtype=bool)
        else:
            self.mask = np.nan_to_num(mask)
        self.pixel_array = pixel_array * self.mask
        self.affine = affine
        self.pixel_spacing = (np.linalg.norm(self.affine[:3, 1]),
                              np.linalg.norm(self.affine[:3, 0]))
        self.function = fitting_function_name
        self.input_list = model_input
        self.convergence = convergence
        self.multithread = multithread
        self.log = log
        if isinstance(elastix_params, dict) == True:
            self.elastix_params = Custom_BSplines(elastix_params)
        elif self.function == "DWI_Moco": self.elastix_params = DWI_BSplines()
        elif self.function == "T1_Moco": self.elastix_params = T1_BSplines()
        # Perform motion correction
        if len(self.shape) == 3:
            output = model_driven_registration(self.pixel_array,
                                               self.pixel_spacing,
                                               fitting_functions,
                                               self.input_list,
                                               self.elastix_params,
                                               precision=self.convergence,
                                               parallel=self.multithread,
                                               function=self.function,
                                               log=self.log)
            self._mdr_results = output
        if len(self.shape) == 4:
            for index in range(self.shape[2]):
                pixel_array_3D = self.pixel_array[: , :, index, :]
                output = model_driven_registration(pixel_array_3D,
                                                   self.pixel_spacing,
                                                   fitting_functions,
                                                   self.input_list,
                                                   self.elastix_params,
                                                   precision=self.convergence,
                                                   parallel=self.multithread,
                                                   function=self.function,
                                                   log=self.log)
                self._mdr_results.append(output)
    
    def get_coregistered(self):
        if len(self.shape) == 3:
            coregistered = np.array(self._mdr_results[0])
        if len(self.shape) == 4:
            coregistered_list = []
            for individual_slice in self._mdr_results:
                coregistered_list.append(individual_slice[0])
            coregistered = np.stack(np.array(coregistered_list), axis=-2)
        return coregistered

    def get_fitted(self):
        if len(self.shape) == 3:
            fitted = np.array(self._mdr_results[1])
        if len(self.shape) == 4:
            fitted_list = []
            for individual_slice in self._mdr_results:
                fitted_list.append(individual_slice[1])
            fitted = np.stack(np.array(fitted_list), axis=-2)
        return fitted

    def get_deformation_field(self):
        if len(self.shape) == 3:
            deformation_field = np.array(self._mdr_results[2])
        if len(self.shape) == 4:
            deformation_list = []
            for individual_slice in self._mdr_results:
                deformation_list.append(individual_slice[2])
            deformation_field = np.stack(np.array(deformation_list), axis=-3)
        return deformation_field

    def get_output_parameters(self):
        if len(self.shape) == 3:
            parameters = np.array(self._mdr_results[3])
        if len(self.shape) == 4:
            parameters_list = []
            for individual_slice in self._mdr_results:
                parameters_list.append(individual_slice[3])
            parameters = np.stack(np.array(parameters_list), axis=-2)
        return parameters

    def get_improvements(self):
        if len(self.shape) == 3:
            improvements = self._mdr_results[4]
        if len(self.shape) == 4:
            improvements = [individual_slice[4] \
                            for individual_slice in self._mdr_results]
        return improvements

    def get_diff_orig_coreg(self):
        difference = self.pixel_array - self.get_coregistered()
        return difference
    
    def to_nifti(self, output_directory=os.getcwd(), base_file_name='Output',
                 maps='all'):
        """Exports masks to NIFTI.

        Parameters
        ----------
        output_directory : string, optional
            Path to the folder where the NIFTI files will be saved.
        base_file_name : string, optional
            Filename of the resulting NIFTI. This code appends the extension.
            Eg., base_file_name = 'Output' will result in 'Output.nii.gz'.
        maps : list or 'all', optional
            List of maps to save to NIFTI. This should either the string "all"
            or a list of maps from ["mask", "coregistered", "difference",
            "fitted", "deformation_field", "parameters"]
        """
        os.makedirs(output_directory, exist_ok=True)
        base_path = os.path.join(output_directory, base_file_name)
        if maps == 'all' or maps == ['all']:
            maps = ['mask', 'original', 'coregistered', 'difference',
                    'fitted', 'deformation_field', 'parameters']
        if isinstance(maps, list):
            for result in maps:
                if result == 'mask':
                    mask_nifti = nib.Nifti1Image(self.mask.astype(int),
                                                 affine=self.affine)
                    nib.save(mask_nifti, base_path + '_mask.nii.gz')
                elif result == 'original':
                    original_nifti = nib.Nifti1Image(self.pixel_array,
                                                     affine=self.affine)
                    nib.save(original_nifti, base_path + '_original.nii.gz')
                elif result == 'coregistered':
                    coreg_nifti = nib.Nifti1Image(self.get_coregistered(),
                                                  affine=self.affine)
                    nib.save(coreg_nifti, base_path + '_coregistered.nii.gz')
                elif result == 'difference':
                    diff_nifti = nib.Nifti1Image(self.get_diff_orig_coreg(),
                                                 affine=self.affine)
                    nib.save(diff_nifti, base_path + '_difference.nii.gz')
                elif result == 'fitted':
                    fit_nifti = nib.Nifti1Image(self.get_fitted(),
                                                affine=self.affine)
                    nib.save(fit_nifti, base_path + '_fitted.nii.gz')
                elif result == 'deformation_field':
                    def_nifti = nib.Nifti1Image(self.get_deformation_field(),
                                                affine=self.affine)
                    nib.save(def_nifti, base_path +
                             '_deformation_field.nii.gz')
                elif result == 'parameters':
                    param_nifti = nib.Nifti1Image(self.get_output_parameters(),
                                                  affine=self.affine)
                    nib.save(param_nifti, base_path + '_parameters.nii.gz')
        else:
            raise ValueError('No NIFTI file saved. The variable "maps" '
                             'should be "all" or a list of maps from '
                             '"["mask", "coregistered", "difference", '
                             '"fitted", "deformation_field", "parameters"]".')