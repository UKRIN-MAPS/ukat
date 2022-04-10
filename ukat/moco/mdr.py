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
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mdreg import MDReg
import mdreg.models as mdl
#from MDR.MDR import model_driven_registration
#from MDR.Tools import export_animation
from ukat.moco.fitting_functions import DWI_Moco, T1_Moco
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
        self.pixel_array = pixel_array
        self.shape = self.pixel_array.shape
        if mask is None:
            self.mask = None
        else:
            self.mask = np.array(np.nan_to_num(mask), dtype=int)
        self.affine = affine
        self.pixel_spacing = (np.linalg.norm(self.affine[:3, 1]),
                              np.linalg.norm(self.affine[:3, 0]))
        self.function = fitting_function_name
        self.input_list = model_input
        self.convergence = convergence
        self.multithread = multithread
        self.log = log
        if self.function == "DWI_Moco":
            self.fitting_function = DWI_Moco
            self.elastix_params = DWI_BSplines()
        elif self.function == "T1_Moco":
            self.fitting_function = T1_Moco
            self.elastix_params = T1_BSplines()
        else:
            self.fitting_function = mdl.constant
        if isinstance(elastix_params, dict) == True:
            self.elastix_params = Custom_BSplines(elastix_params)
        
        # Perform motion correction
        if len(self.shape) == 3:
            mdr = MDReg()
            mdr.set_array(self.pixel_array)
            mdr.set_mask(self.mask)
            mdr.pixel_spacing = self.pixel_spacing
            mdr.signal_model = self.fitting_function
            mdr.signal_parameters = self.input_list
            mdr.elastix = self.elastix_params
            mdr.convergence = self.convergence
            mdr.parallel = self.multithread
            mdr.log = self.log
            mdr.fit()
            self._mdr_results = mdr
        if len(self.shape) == 4:
            for index in range(self.shape[2]):
                pixel_array_3D = self.pixel_array[: , :, index, :]
                mask_3D = self.mask[: , :, index, :]
                mdr = MDReg()
                mdr.set_array(pixel_array_3D)
                mdr.set_mask(mask_3D)
                mdr.pixel_spacing = self.pixel_spacing
                mdr.signal_model = self.fitting_function
                mdr.signal_parameters = self.input_list
                mdr.elastix = self.elastix_params
                mdr.convergence = self.convergence
                mdr.parallel = self.multithread
                mdr.log = self.log
                mdr.fit()
                self._mdr_results.append(mdr)
    
    def get_coregistered(self):
        # mdr.coreg
        if isinstance(self._mdr_results, list):
            coregistered_list = []
            for individual_slice in self._mdr_results:
                coregistered_list.append(individual_slice.coreg)
            coregistered = np.stack(np.array(coregistered_list), axis=-2)
        else:
            coregistered = np.array(self._mdr_results.coreg)
        return coregistered

    def get_model_fit(self):
        # mdr.model_fit
        if isinstance(self._mdr_results, list):
            model_fit_list = []
            for individual_slice in self._mdr_results:
                model_fit_list.append(individual_slice.model_fit)
            model_fit = np.stack(np.array(model_fit_list), axis=-2)
        else:
            model_fit = np.array(self._mdr_results.model_fit)
        return model_fit

    def get_deformation_field(self):
        # mdr.deformation
        if isinstance(self._mdr_results, list):
            deformation_list = []
            for individual_slice in self._mdr_results:
                deformation_list.append(individual_slice.deformation)
            deformation_field = np.stack(np.array(deformation_list), axis=-3)
        else:
            deformation_field = np.array(self._mdr_results.deformation)
        return deformation_field

    def get_output_parameters(self):
        # mdr.pars
        if isinstance(self._mdr_results, list):
            parameters_list = []
            for individual_slice in self._mdr_results:
                parameters_list.append(individual_slice.pars)
            parameters = np.stack(np.array(parameters_list), axis=-2)
        else:
            parameters = np.array(self._mdr_results.pars)
        return parameters

    def get_improvements(self, export=False, output_directory=os.getcwd()):
        # mdr.iter
        if isinstance(self._mdr_results, list):
            improvements = []
            for index, individual_slice in self._mdr_results:
                improvements.append(individual_slice.iter)
                if export:
                    individual_slice.iter.to_csv(os.path.join(output_directory,
                                                 "improvements_slice_" + \
                                                 str(index) + ".csv"))
        else:
            improvements = self._mdr_results.iter
            if export:
                improvements.to_csv(os.path.join(output_directory,
                                                 "improvements.csv"))
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
            "model_fit", "deformation_field", "parameters"]
        """
        os.makedirs(output_directory, exist_ok=True)
        base_path = os.path.join(output_directory, base_file_name)
        if maps == 'all' or maps == ['all']:
            maps = ['mask', 'original', 'coregistered', 'difference',
                    'model_fit', 'deformation_field', 'parameters']
        if isinstance(maps, list):
            for result in maps:
                if result == 'mask' and self.mask is not None:
                    mask_nifti = nib.Nifti1Image(self.mask[..., 0].astype(int),
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
                elif result == 'model_fit':
                    fit_nifti = nib.Nifti1Image(self.get_model_fit(),
                                                affine=self.affine)
                    nib.save(fit_nifti, base_path + '_model_fit.nii.gz')
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
                             '"model_fit", "deformation_field", '
                             '"parameters"]".')

    def to_gif(self, output_directory=os.getcwd(), base_file_name='Output',
               slice_number=None, maps='all'):
        """Exports masks to GIF.

        Parameters
        ----------
        output_directory : string, optional
            Path to the folder where the GIF files will be saved.
        base_file_name : string, optional
            Filename of the resulting GIF. This code appends the extension.
            Eg., base_file_name = 'Output' will result in 'Output.gif'.
        slice_number : int, optional
            dfsdfgdsgdsgdsg
        maps : list or 'all', optional
            List of maps to save to GIF. This should either the string "all"
            or a list of maps from ["mask", "coregistered", "difference",
            "model_fit", "deformation_field", "parameters"]
        """
        os.makedirs(output_directory, exist_ok=True)
        if maps == 'all' or maps == ['all']:
            maps = ['mask', 'original', 'coregistered', 'difference',
                    'model_fit', 'deformation_field', 'parameters']
        if isinstance(maps, list):
            for result in maps:
                if result == 'mask':
                    array = self.mask.astype(int)
                    file_name = base_file_name + '_mask'
                elif result == 'original':
                    array = self.pixel_array
                    file_name = base_file_name + '_original'
                elif result == 'coregistered':
                    array = self.get_coregistered()
                    file_name = base_file_name + '_coregistered'
                elif result == 'difference':
                    array = self.get_diff_orig_coreg()
                    file_name = base_file_name + '_difference'
                elif result == 'model_fit':
                    array = self.get_model_fit()
                    file_name = base_file_name + '_model_fit'
                elif result == 'deformation_field':
                    array = self.get_deformation_field()
                    # Merge the last 2 dimensions. 
                    # The penultimate dimension (position -2) has length = 2.
                    array = np.squeeze(np.reshape(array, array.shape[:-2] +
                                                  (-1, array.shape[-2] *
                                                   array.shape[-1])))
                    file_name = base_file_name + '_deformation_field'
                elif result == 'parameters':
                    array = self.get_output_parameters()
                    file_name = base_file_name + '_parameters'
                if len(np.shape(array)) == 4:
                    if ~isinstance(slice_number, int):
                        slice_number = int(np.shape(array)[2] / 2)
                    array = array[:, :, slice_number, :]
                array = np.rot90(array)
                fig = plt.figure()
                im = plt.imshow(np.squeeze(array[:,:,0]), animated=True)
                def updatefig(i):
                    im.set_array(np.squeeze(array[:,:,i]))
                anim = animation.FuncAnimation(fig, updatefig,interval=50,
                                               frames=array.shape[2])
                anim.save(os.path.join(output_directory, file_name + '.gif'))
        else:
            raise ValueError('No GIF file saved. The variable "maps" '
                             'should be "all" or a list of maps from '
                             '"["mask", "coregistered", "difference", '
                             '"model_fit", "deformation_field", '
                             '"parameters"]".')
