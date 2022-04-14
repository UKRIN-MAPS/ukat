"""
The MotionCorrection class uses the `mdreg` Python Package
(https://github.com/QIB-Sheffield/mdreg),
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
from ukat.moco.fitting_functions import DWI_Moco, T1_Moco
from ukat.moco.elastix_parameters import (DWI_BSplines, T1_BSplines,
                                          Custom_Parameter_Map)


class MotionCorrection:
    """
    Performs Model-Driven Registration in a volume or series of volumes.
    """

    def __init__(self, pixel_array, affine, fitting_function_name, model_input,
                 mask=None, convergence=1.0, elastix_params=None,
                 multithread=True, log=True):
        """Initialise a MotionCorrection class instance.

        Parameters
        ----------
        pixel_array : np.ndarray
            Array containing the MRI volume(s) acquired at different
            time points. Its expected shape should be 3 or 4.
        affine : np.ndarray
            A matrix giving the relationship between voxel coordinates and
            world coordinates.
        fitting_function_name : string
            String containing the name of the model fit python class.
        model_input : list
            List containing the input arguments of the model fit python class.
        mask : np.ndarray, optional
            A boolean mask of the voxels to fit. It must have the same shape as
            the 'pixel_array'.
        convergence : float, optional
            Stopping criteria value, i.e. the condition that must be reached
            in order to stop the execution of the algorithm. If the improvement
            value after co-registration is lower than the convergence value,
            then the model-driven registration is concluded.
        elastix_params : dict, optional
            If a dictionary is provided, then an elastix registration parameter
            object will be created based on the dictionary and used in the
            model-driven motion correction process.
        multithread : bool, optional
            If True (default), co-registration by `itk-elastix` will be
            distributed over all cores available on the node.
        log : bool, optional
            If True (default), the `itk-elastix` co-registration
            process output is printed in the terminal and
            saved to a text file.
        """
        self._mdr_results = []
        self.pixel_array = pixel_array
        self.shape = self.pixel_array.shape
        if mask is None:
            self.mask = np.ones(np.shape(self.pixel_array), dtype=int)
        else:
            self.mask = np.array(np.nan_to_num(mask), dtype=int)
        self.affine = affine
        self._pixel_spacing = (np.linalg.norm(self.affine[:3, 1]),
                               np.linalg.norm(self.affine[:3, 0]))
        self.function = fitting_function_name
        self.model_input = model_input
        self.convergence = convergence
        self.multithread = multithread
        self.log = log
        if self.function == "DWI_Moco":
            self._fitting_function = DWI_Moco
            self._elastix_params = DWI_BSplines()
        elif self.function == "T1_Moco":
            self._fitting_function = T1_Moco
            self._elastix_params = T1_BSplines()
        else:
            self._fitting_function = mdl.constant
            self._elastix_params = Custom_Parameter_Map({}, 'rigid')
        # The user may provide a python dictionary with the elastix parameters.
        if isinstance(elastix_params, dict):
            self._elastix_params = Custom_Parameter_Map(elastix_params)

        # Perform motion correction
        if len(self.shape) == 3:
            mdr = MDReg()
            mdr.set_array(self.pixel_array)
            mdr.set_mask(self.mask)
            mdr.pixel_spacing = self._pixel_spacing
            mdr.signal_model = self._fitting_function
            mdr.signal_parameters = self.model_input
            mdr.elastix = self._elastix_params
            mdr.convergence = self.convergence
            mdr.parallel = self.multithread
            mdr.log = self.log
            mdr.fit()
            self._mdr_results = mdr
        elif len(self.shape) == 4:
            for index in range(self.shape[2]):
                pixel_array_3D = self.pixel_array[:, :, index, :]
                mask_3D = self.mask[:, :, index, :]
                mdr = MDReg()
                mdr.set_array(pixel_array_3D)
                mdr.set_mask(mask_3D)
                mdr.pixel_spacing = self._pixel_spacing
                mdr.signal_model = self._fitting_function
                mdr.signal_parameters = self.model_input
                mdr.elastix = self._elastix_params
                mdr.convergence = self.convergence
                mdr.parallel = self.multithread
                mdr.log = self.log
                mdr.fit()
                self._mdr_results.append(mdr)

    def get_coregistered(self):
        """
        Returns the motion corrected images from
        the provided input 'pixel_array'.

        Returns
        -------
        coregistered : np.ndarray
            The co-registered image array.
        """
        if isinstance(self._mdr_results, list):
            coregistered_list = []
            for individual_slice in self._mdr_results:
                coregistered_list.append(individual_slice.coreg)
            coregistered = np.stack(np.array(coregistered_list), axis=-2)
        else:
            coregistered = np.array(self._mdr_results.coreg)
        return coregistered

    def get_diff_orig_coreg(self):
        """
        Returns the difference between 'pixel_array' and 'coregistered'.

        Returns
        -------
        difference : np.ndarray
            The difference image array between the input image and the motion
            corrected image.
        """
        difference = self.pixel_array - self.get_coregistered()
        return difference

    def get_model_fit(self):
        """
        Returns the signal model fit images from
        the provided input 'pixel_array'.

        Returns
        -------
        model_fit : np.ndarray
            The signal model fit image array.
        """
        if isinstance(self._mdr_results, list):
            model_fit_list = []
            for individual_slice in self._mdr_results:
                model_fit_list.append(individual_slice.model_fit)
            model_fit = np.stack(np.array(model_fit_list), axis=-2)
        else:
            model_fit = np.array(self._mdr_results.model_fit)
        return model_fit

    def get_deformation_field(self):
        """
        Returns the deformation field that results from the co-registration
        part of the model-driven registration process. The shape should be
        the same as 'pixel_array' + (,2).

        Returns
        -------
        deformation_field : np.ndarray
            The deformation field image array.
        """
        if isinstance(self._mdr_results, list):
            deformation_list = []
            for individual_slice in self._mdr_results:
                deformation_list.append(individual_slice.deformation)
            deformation_field = np.stack(np.array(deformation_list), axis=-3)
        else:
            deformation_field = np.array(self._mdr_results.deformation)
        return deformation_field

    def get_parameters(self):
        """
        Returns the fitted parameters of the signal model fit from
        the provided input 'pixel_array'.

        Returns
        -------
        parameters : list
            List containing the output fitted parameter images.
        """
        if isinstance(self._mdr_results, list):
            parameters_list = []
            for individual_slice in self._mdr_results:
                parameters_list.append(individual_slice.pars)
            parameters = np.stack(np.array(parameters_list), axis=-2)
        else:
            parameters = np.array(self._mdr_results.pars)
        return parameters

    def get_improvements(self, output_directory=os.getcwd(),
                         base_file_name='improvements',
                         export=False):
        """
        Returns a pandas DataFrame and it represents the maximum deformation
        per co-registration iteration calculated as the euclidean distance of
        difference between old and new deformation field in each iteration.

        Parameters
        ----------
        output_directory : string, optional
            Path to the folder that will contain the CSV file to be saved,
            if export=True.
        base_file_name : string, optional
            Filename of the exported CSV. This code appends the extension.
            Eg., base_file_name = 'improvements' will result in
            'improvements.csv'.
        export : bool, optional
            If True (default is False), the table with the improvements values
            is saved as "improvements.csv" or "improvements_slice_x.csv" if the
            images are volumetric in the 'output_directory'.

        Returns
        -------
        improvements : pandas.DataFrame
            The table with the improvements values per iteration. The last
            value of the table should be lower than self.convergence.
        """
        if isinstance(self._mdr_results, list):
            improvements = []
            for index, individual_slice in enumerate(self._mdr_results):
                improvements.append(individual_slice.iter)
                if export:
                    file_path = os.path.join(output_directory,
                                             base_file_name + "_slice_" +
                                             str(index) + ".csv")
                    individual_slice.iter.to_csv(file_path)
        else:
            improvements = self._mdr_results.iter
            if export:
                file_path = os.path.join(output_directory,
                                         base_file_name + ".csv")
                improvements.to_csv(file_path)
        return improvements

    def get_elastix_parameters(self, output_directory=os.getcwd(),
                               base_file_name='Elastix_Parameters',
                               export=False):
        """
        Returns a itk.ParameterObject with the elastix registration parameters.

        Parameters
        ----------
        output_directory : string, optional
            Path to the folder that will contain the TXT file to be saved,
            if export=True.
        base_file_name : string, optional
            Filename of the exported TXT. This code appends the extension.
            Eg., base_file_name = 'Elastix_Parameters' will result in
            'Elastix_Parameters.txt'.
        export : bool, optional
            If True (default is False), the elastix registration parameters
            are saved as "Elastix_Parameters.txt" in the 'output_directory'.

        Returns
        -------
        self._elastix_params : itk.ParameterObject
            Private attribute with the elastix registration parameters
            that is returned if this getter function is called.
        """
        if export:
            file_path = os.path.join(output_directory,
                                     base_file_name + ".txt")
            text_file = open(file_path, "w")
            print(self._elastix_params, file=text_file)
            text_file.close()
        return self._elastix_params

    def to_nifti(self, output_directory=os.getcwd(), base_file_name='Output',
                 maps='all'):
        """Exports some of the MotionCorrection class results to NIFTI.

        Parameters
        ----------
        output_directory : string, optional
            Path to the folder where the NIFTI files will be saved.
        base_file_name : string, optional
            Filename of the resulting NIFTI. This code appends the extension.
            Eg., base_file_name = 'Output' will result in 'Output.nii.gz'.
        maps : list or 'all', optional
            List of maps to save to NIFTI. This should either the string "all"
            or a list of maps from ["mask", "original", "coregistered",
            "difference", "model_fit", "deformation_field", "parameters"]
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
                    param_nifti = nib.Nifti1Image(self.get_parameters(),
                                                  affine=self.affine)
                    nib.save(param_nifti, base_path + '_parameters.nii.gz')
        else:
            raise ValueError('No NIFTI file saved. The variable "maps" '
                             'should be "all" or a list of maps from '
                             '"["mask", "original", "coregistered", '
                             '"difference", "model_fit", "deformation_field", '
                             '"parameters"]".')

    def to_gif(self, output_directory=os.getcwd(), base_file_name='Output',
               slice_number=None, maps='all'):
        """Exports some of the MotionCorrection class results to GIF.

        Parameters
        ----------
        output_directory : string, optional
            Path to the folder where the GIF files will be saved.
        base_file_name : string, optional
            Filename of the resulting GIF. This code appends the extension.
            Eg., base_file_name = 'Output' will result in 'Output.gif'.
        slice_number : int, optional
            If the image is volumetric, this value is the index of the
            slice chosen for the GIF animation. The central slice
            is the one used by default.
        maps : list or 'all', optional
            List of maps to save to GIF. This should either the string "all"
            or a list of maps from ["mask", "original", "coregistered",
            "difference", "model_fit", "deformation_field", "parameters"]
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
                    array = self.get_parameters()
                    file_name = base_file_name + '_parameters'
                if len(np.shape(array)) == 4:
                    if ~isinstance(slice_number, int):
                        slice_number = int(np.shape(array)[2] / 2)
                    array = array[:, :, slice_number, :]
                array = np.rot90(array)
                fig = plt.figure()
                im = plt.imshow(np.squeeze(array[:, :, 0]), animated=True)

                def updatefig(i):
                    im.set_array(np.squeeze(array[:, :, i]))
                anim = animation.FuncAnimation(fig, updatefig, interval=50,
                                               frames=array.shape[2])
                anim.save(os.path.join(output_directory, file_name + '.gif'))
        else:
            raise ValueError('No GIF file saved. The variable "maps" '
                             'should be "all" or a list of maps from '
                             '"["mask", "original", "coregistered", '
                             '"difference", "model_fit", "deformation_field", '
                             '"parameters"]".')
