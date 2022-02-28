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
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tabulate import tabulate
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
    num_pixels : np.ndarray
        List containing the number of True values in the mask per phase.
    area : np.ndarray
        List containing the area (cm2) of the mask per phase.
    min_velocity : np.ndarray
        List containing the minimum velocity values (cm/s) per phase.
    mean_velocity : np.ndarray
        List containing the average velocity values (cm/s) per phase.
    max_velocity : np.ndarray
        List containing the maximum velocity values (cm/s) per phase.
    std_velocity : np.ndarray
        List containing the std dev of the velocity values (cm/s) per phase.
    rbf : np.ndarray
        List containing the Renal Blood Flow values (ml/min) per phase.
    mean_velocity_global : float
        Average velocity (cm/s) across the different phases.
    mean_rbf : float
        Average Renal Blood Flow (ml/min) across the different phases.
    resistive_index : float
        A prognostic marker in renal vascular diseases which range is [0, 1].
    """

    def __init__(self, velocity_array, affine, mask):
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
        mask : np.ndarray
            A boolean mask of the voxels to fit. Should be the shape of
            the raw data.
        """
        self.shape = velocity_array.shape
        self.affine = affine
        self.pixel_spacing = (np.linalg.norm(self.affine[:3, 1]),
                              np.linalg.norm(self.affine[:3, 0]))
        self.mask = np.where(mask, mask, np.nan)
        self.velocity_array = np.abs(velocity_array * self.mask)
        self.num_pixels = []
        self.area = []
        self.min_velocity = []
        self.mean_velocity = []
        self.max_velocity = []
        self.std_velocity = []
        self.rbf = []
        self.mean_velocity_global = 0
        self.mean_rbf = 0
        self.resistive_index = 0
        if len(self.shape) == 3:
            # Extract number pixels, area, velocity stats (min, mean, max, std)
            # and renal blood flow (RBF)
            self.num_pixels = np.count_nonzero(~np.isnan(self.velocity_array),
                                               axis=(0, 1))
            # area = (num_pixels * mm * mm * 0.01) = cm2
            self.area = (self.num_pixels * self.pixel_spacing[0] *
                         self.pixel_spacing[1] * 0.01)
            self.min_velocity = np.nanmin(self.velocity_array, axis=(0, 1))
            self.mean_velocity = np.nanmean(self.velocity_array, axis=(0, 1))
            self.max_velocity = np.nanmax(self.velocity_array, axis=(0, 1))
            self.std_velocity = np.nanstd(self.velocity_array, axis=(0, 1))
            # q = (60 * cm2 * cm/s) = ml/min
            self.rbf = 60 * np.array(self.area * self.mean_velocity)
            # Mean velocity global and mean flow
            self.mean_velocity_global = np.mean(self.mean_velocity)
            self.mean_rbf = np.mean(self.rbf)
            # Restrictive Index
            mean_velocity_systole = np.max(self.mean_velocity)
            mean_velocity_diastole = np.min(self.mean_velocity)
            self.resistive_index = ((mean_velocity_systole -
                                     mean_velocity_diastole) /
                                    mean_velocity_systole)
            # Convert any nan values to 0
            self.velocity_array = np.nan_to_num(self.velocity_array)
            self.mask = np.nan_to_num(self.mask)
        else:
            raise ValueError('The input velocity_array should be 3D.')

    def get_stats_table(self):
        """
        Stores most of PhaseContrast class attributes into a pandas DataFrame.

        Returns
        ----------
        table : pandas.DataFrame
            Returns a table with the results/stats of each output per phase.
        """
        stats = {"RBF (ml/min)": self.rbf,
                 "Area (cm2)": self.area,
                 "Nr Pixels": self.num_pixels,
                 "Mean Vel (cm/s)": self.mean_velocity,
                 "Min Vel (cm/s)": self.min_velocity,
                 "Max Vel (cm/s)": self.max_velocity,
                 "StdDev Vel (cm/s)": self.std_velocity}
        table = pd.DataFrame(data=stats)
        return table

    def print_stats(self):
        """
        Prints the table with the stats for each output per phase.
        """
        stats_table = self.get_stats_table()
        print(tabulate(stats_table, headers='keys', tablefmt='github',
              floatfmt='.3f'))

    def to_csv(self, path):
        """
        Saves the stats_table into a csv file.

        Parameters
        ----------
        path : str
            Path to the desired csv file.
        """
        stats_table = self.get_stats_table()
        stats_table.to_csv(path)

    def plot(self, stat='default', file_name=None):
        """
        This method plots the output PhaseContrast stats per phase.

        Parameters
        ----------
        stat : str, optional
            Name of the output stat variable. This method plots mean velocity
            and the RBF by default.
        file_name : str, optional
            Path to the image file (*.jpg, *.png, etc.)
            in which the plot will be saved.
        """
        if stat == 'default':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            ax1.plot(self.mean_velocity, 'ro-')
            ax1.set_ylabel('Velocity (cm/sec)')
            ax1.set_xlabel('Phase')
            ax1.set_title('Average Velocity')
            ax2.plot(self.rbf, 'b-')
            ax2.set_ylabel('RBF (ml/min)')
            ax2.set_xlabel('Phase')
            ax2.set_title('Renal Artery Blood Flow')
        else:
            if stat == 'min_velocity':
                stat_variable = self.min_velocity
                y_label = 'Velocity (cm/sec)'
                title = 'Minimum Velocity'
            elif stat == 'mean_velocity':
                stat_variable = self.mean_velocity
                y_label = 'Velocity (cm/sec)'
                title = 'Average Velocity'
            elif stat == 'max_velocity':
                stat_variable = self.max_velocity
                y_label = 'Velocity (cm/sec)'
                title = 'Maximum Velocity'
            elif stat == 'std_velocity':
                stat_variable = self.std_velocity
                y_label = 'Velocity (cm/sec)'
                title = 'Standard Deviation of the Velocity'
            elif stat == 'rbf':
                stat_variable = self.rbf
                y_label = 'RBF (ml/min)'
                title = 'Renal Artery Blood Flow'
            elif stat == 'num_pixels':
                stat_variable = self.num_pixels
                y_label = '# Pixels'
                title = 'Number of Pixels in the Region Of Interest (ROI)'
            elif stat == 'area':
                stat_variable = self.area
                y_label = 'Area ($cm^2$)'
                title = 'Area of the Region Of Interest (ROI)'
            else:
                raise ValueError('The stat provided is not valid.')
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.plot(stat_variable, 'ro-')
            ax.set_ylabel(y_label)
            ax.set_xlabel('Phase')
            ax.set_title(title)
        # The following saves the plot(s) to file_name, if given.
        if file_name is not None:
            fig.savefig(file_name)

    def to_nifti(self, output_directory=os.getcwd(), base_file_name='Output',
                 maps='all'):
        """Exports the velocity array and the renal artery mask to NIfTI.

        Parameters
        ----------
        output_directory : string, optional
            Path to the folder where the NIfTI files will be saved.
        base_file_name : string, optional
            Filename of the resulting NIfTI. This code appends the extension.
            Eg., base_file_name = 'Output' will result in 'Output.nii.gz'.
        maps : list or 'all', optional
            List of maps to save to NIfTI. This should either the string "all"
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
            raise ValueError('No NIfTI file saved. The variable "maps" '
                             'should be "all" or a list of maps from '
                             '"["velocity_array", "mask"]".')


def convert_to_velocity(pixel_array, velocity_encoding,
                        velocity_encode_scale=None):
    """
    Calculate the velocity array from the given input image and
    velocity encoding. If a velocity encode scale is given then it is
    used to convert the pixel_value to radians.

    Parameters
    ----------
    pixel_array: np.ndarray
        A 3D array containing the phase images of the phase contrast sequence.
    velocity_encoding : float
        The value of the velocity encoding in cm/s.
    velocity_encode_scale : float, optional
        If given, this value is used to scale from image intensity to radians.

    Returns
    -------
    velocity_array: np.ndarray
        A 3D array containing the velocity images of the phase contrast
        sequence, i.e. the dimensions of the array are [x, y, p], where p
        corresponds to the phase (or trigger delay).
    """
    if velocity_encode_scale is not None:
        pi_range_array = pixel_array / velocity_encode_scale
    else:
        pi_range_array = convert_to_pi_range(pixel_array)
    velocity_array = pi_range_array * velocity_encoding
    return velocity_array
