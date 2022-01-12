"""
The velocity and flow calculations in this Phase Contrast class
are based on the scientific paper:

"Phase-contrast magnetic resonance imaging to assess renal perfusion:
a systematic review and statement paper"

Giulia Villa, Steffen Ringgaard, Ingo Hermann, Rebecca Noble, Paolo Brambilla,
Dinah S. Khatir, Frank G. Zöllner, Susan T. Francis, Nicholas M. Selby,
Andrea Remuzzi, Anna Caroli

Magnetic Resonance Materials in Physics, Biology and Medicine (2020) 33:3-21
https://doi.org/10.1007/s10334-019-00772-0
"""

import os
import csv
import numpy as np
import nibabel as nib
from ukat.utils.tools import convert_to_pi_range


class PhaseContrast:
    """
    Generates velocity and flow measurements of an MRI Phase Contrast Sequence.

    Attributes
    ----------
    velocity_array : np.ndarray
        The input velocity array masked.
    shape : tuple
        The shape of the velocity array.
    pixel_spacing : tuple
        The pixel spacing of the acquisition estimated from the affine.
    mask : np.ndarray
        A boolean mask of the voxels to fit.
    num_pixels_phase : list
        List containing the number of True values in the mask per phase.
    area_phase : list
        List containing the area (cm2) of the mask per phase.
    min_velocity_phase : list
        List containing the minimum velocity values (cm/s) per phase.
    mean_velocity_phase : list
        List containing the average velocity values (cm/s) per phase.
    max_velocity_phase : list
        List containing the maximum velocity values (cm/s) per phase.
    peak_velocity_phase : list
        List containing the maximum velocity values (cm/s) per phase.
    std_velocity_phase : list
        List containing the std dev of the velocity values (cm/s) per phase.
    RBF : list
        List containing the Renal Blood Flow values (ml/min) per phase.
    stats_table : dictionary
        A dictionary containing all class attributes that are a list.
    mean_velocity : float
        Average velocity (cm/s) accross the different phases.
    mean_RBF : float
        Average Renal Blood Flow (ml/min) accross the different phases.
    resistive_index : float
        A prognostic marker in renal vascular diseases which range is [0, 1].
    """

    def __init__(self, velocity_array, affine, mask=None):
        """Initialise a PhaseContrast class instance.

        Parameters
        ----------
        velocity_array : np.ndarray
            A 3D array containing the velocity images of the phase contrast
            sequence, i.e. the dimensions of the array are [x, y, p], where p
            corresponds to the phase (or trigger delay).
        affine : np.ndarray
            A matrix giving the relationship between voxel coordinates and
            world coordinates.
        mask : np.ndarray, optional
            A boolean mask of the voxels to fit. Should be the shape of the
            the raw data.
        """
        self.shape = velocity_array.shape
        self.affine = affine
        self.pixel_spacing = (np.linalg.norm(self.affine[:3, 1]),
                              np.linalg.norm(self.affine[:3, 0]))
        # Generate a mask if there isn't one specified
        if mask is None:
            self.mask = np.ones(self.shape, dtype=bool)
        else:
            self.mask = np.where(mask == 0, np.nan, mask)
            # The purpose is for the np.nanmean to work properly later
            # without including the 0s of the mask
        self.velocity_array = np.abs(velocity_array * self.mask)
        self.num_pixels_phase = []
        self.area_phase = []
        self.min_velocity_phase = []
        self.mean_velocity_phase = []
        self.max_velocity_phase = []
        self.peak_velocity_phase = []
        self.std_velocity_phase = []
        self.RBF = []
        self.mean_velocity = 0
        self.mean_RBF = 0
        self.resistive_index = 0
        if len(self.shape) == 3:
            for phase in range(self.shape[-1]):
                phase_array = self.velocity_array[..., phase]
                num_pixels = np.count_nonzero(~np.isnan(phase_array))
                area = (num_pixels * 0.1 * self.pixel_spacing[0] *
                        0.1 * self.pixel_spacing[1])  # (0.1*mm * 0.1*mm) = cm2
                min_vel = np.nanmin(phase_array)
                avrg_vel = np.nanmean(phase_array)
                max_vel = np.nanmax(phase_array)
                std_vel = np.nanstd(phase_array)
                Q = 60 * area * avrg_vel  # (60s*cm2*cm/s) = cm3/min = ml/min
                self.num_pixels_phase.append(num_pixels)
                self.area_phase.append(area)
                self.min_velocity_phase.append(min_vel)
                self.mean_velocity_phase.append(avrg_vel)
                self.max_velocity_phase.append(max_vel)
                self.peak_velocity_phase.append(max_vel)
                self.std_velocity_phase.append(std_vel)
                self.RBF.append(Q)
            # Build table with the results per phase (list of lists)
            self.stats_table = {"Phase": list(np.arange(self.shape[-1])),
                                "RBF (ml/min)": self.RBF,
                                "Area (cm2)": self.area_phase,
                                "Nr Pixels": self.num_pixels_phase,
                                "Mean Vel (cm/s)": self.mean_velocity_phase,
                                "Min Vel (cm/s)": self.min_velocity_phase,
                                "Max Vel (cm/s)": self.max_velocity_phase,
                                "Peak Vel (cm/s)": self.peak_velocity_phase,
                                "StdDev Vel (cm/s)": self.std_velocity_phase}
            # Mean velocity and mean flow
            self.mean_velocity = np.mean(self.mean_velocity_phase)
            self.mean_RBF = np.mean(self.RBF)
            # Restrictive Index
            mean_velocity_systole = np.amax(self.mean_velocity_phase)
            mean_velocity_diastole = np.amin(self.mean_velocity_phase)
            self.resistive_index = ((mean_velocity_systole -
                                     mean_velocity_diastole) /
                                    mean_velocity_systole)
            # Convert any nan values to 0
            self.velocity_array = np.nan_to_num(self.velocity_array)
            self.mask = np.nan_to_num(self.mask)
        else:
            raise ValueError('The input velocity_array should be 3D.')

    def save_output_csv(self, path):
        """
        Save most of PhaseContrast class attributes into a csv file.

        Parameters
        ----------
        path : str
            Path to the desired csv file.
        """
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, self.stats_table.keys())
            w.writeheader()
            for index in range(self.shape[-1]):
                temp_dict = {}
                for key in self.stats_table.keys():
                    temp_dict[key] = self.stats_table[key][index]
                w.writerow(temp_dict)
        del temp_dict

    def to_nifti(self, output_directory=os.getcwd(), base_file_name='Output',
                 maps='all'):
        """Exports the velocity array and the renal artery mask to NIFTI.

        Parameters
        ----------
        output_directory : string, optional
            Path to the folder where the NIFTI files will be saved.
        base_file_name : string, optional
            Filename of the resulting NIFTI. This code appends the extension.
            Eg., base_file_name = 'Output' will result in 'Output.nii.gz'.
        maps : list or 'all', optional
            List of maps to save to NIFTI. This should either the string "all"
            or a list of maps from ["phase_array", "mask", "velocity_array"].
        """
        os.makedirs(output_directory, exist_ok=True)
        base_path = os.path.join(output_directory, base_file_name)
        if maps == 'all' or maps == ['all']:
            maps = ["velocity_array", "mask"]
        if isinstance(maps, list):
            for result in maps:
                if result == 'velocity_array' or result == 'velocity array':
                    velocity_nifti = nib.Nifti1Image(self.velocity_array,
                                                     affine=self.affine)
                    nib.save(velocity_nifti, base_path +
                             '_velocity_array.nii.gz')
                elif result == 'mask':
                    mask_nifti = nib.Nifti1Image(self.mask.astype(int),
                                                 affine=self.affine)
                    nib.save(mask_nifti, base_path + '_mask.nii.gz')
        else:
            raise ValueError('No NIFTI file saved. The variable "maps" '
                             'should be "all" or a list of maps from '
                             '"["velocity_array", "mask"]".')


def convert_to_velocity(image_array, velocity_encoding):
    """
    Calculate the velocity array from the given input phase image and
    velocity encoding value.

    Parameters
    ----------
    image_array: np.ndarray
        A 3D array containing the phase images of the phase contrast sequence.
    velocity_encoding : float
        The value of the velocity encoding in cm/s.

    Returns
    -------
    velocity_array: np.ndarray
        A 3D array containing the velocity images of the phase contrast
        sequence, i.e. the dimensions of the array are [x, y, p], where p
        corresponds to the phase (or trigger delay).
    """
    # velocity_encoding_expected =~ 20cm/s
    return convert_to_pi_range(image_array) * velocity_encoding