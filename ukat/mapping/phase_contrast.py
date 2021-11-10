"""
The velocity and flow calculations in this Phase Contrast class
are based on the scientific paper:

"Phase‑contrast magnetic resonance imaging to assess renal perfusion:
a systematic review and statement paper"

Giulia Villa, Steffen Ringgaard, Ingo Hermann, Rebecca Noble, Paolo Brambilla,
Dinah S. Khatir, Frank G. Zöllner, Susan T. Francis, Nicholas M. Selby,
Andrea Remuzzi, Anna Caroli

Magnetic Resonance Materials in Physics, Biology and Medicine (2020) 33:3–21
https://doi.org/10.1007/s10334-019-00772-0
"""

import os
import numpy as np
import nibabel as nib
from skimage.restoration import unwrap_phase
from ukat.utils.tools import convert_to_pi_range


class PhaseContrast:
    """
    Generates velocity and flow measurements of an MRI Phase Contrast Sequence.

    Attributes
    ----------
    velocity_array : np.ndarray
        The input velocity array masked.
    shape : tuple
        The shape of the phase and velocity arrays.
    pixel_spacing : list
        The pixel spacing of the acquisition estimated from the affine.
    mask : np.ndarray
        A boolean mask of the voxels to fit.
    mean_velocity_cardiac_cycle : list
        List containing the average velocity values (cm/s) per cardiac cycle.
    peak_velocity_cardiac_cycle : list
        List containing the maximum velocity values (cm/s) per cardiac cycle.
    RBF : list
        List containing the Renal Blood Flow values (ml/min) per cardiac cycle.
    mean_velocity : float
        Average velocity (cm/s) accross the cardiac cycles.
    peak_velocity : float
        Peak velocity (cm/s) accross the cardiac cycles.
    mean_RBF : float
        Average Renal Blood Flow (ml/min) accross the cardiac cycles.
    """

    def __init__(self, velocity_array, affine, mask=None):
        """Initialise a PhaseContrast class instance.

        Parameters
        ----------
        velocity_array : np.ndarray
            A 3D array containing the velocity images of the phase contrast
            sequence, i.e. the dimensions of the array are [x, y, c], where c
            corresponds to the cardica cycle.
        affine : np.ndarray
            A matrix giving the relationship between voxel coordinates and
            world coordinates.
        mask : np.ndarray, optional
            A boolean mask of the voxels to fit. Should be the shape of the
            the raw data.
        unwrap : boolean, optional
            By default, this script applies the
            scipy phase unwrapping for each phase echo image.
        wrap_around : boolean, optional
            By default, this flag from unwrap_phase is False.
            The algorithm will regard the edges along the corresponding axis
            of the image to be connected and use this connectivity to guide the
            phase unwrapping process.Eg., voxels [0, :, :] are considered to be
            next to voxels [-1, :, :] if wrap_around=True.
        """
        self.shape = velocity_array.shape
        self.affine = affine
        self.pixel_spacing = [np.linalg.norm(self.affine[:3, 1]),
                              np.linalg.norm(self.affine[:3, 0])]
        # Generate a mask if there isn't one specified
        if mask is None:
            self.mask = np.ones(self.shape, dtype=bool)
        else:
            self.mask = np.where(mask == 0, np.nan, mask)
            # The purpose is for the np.nanmean to work properly later
            # without including the 0s of the mask
        self.velocity_array = velocity_array * self.mask
        self.mean_velocity_cardiac_cycle = []
        self.peak_velocity_cardiac_cycle = []
        self.RBF = []
        self.mean_velocity = 0
        self.peak_velocity = 0
        self.mean_RBF = 0

        if len(self.shape) == 3:
            for cardiac_cycle in range(self.shape[-1]):
                cardiac_cycle_array = self.velocity_array[..., cardiac_cycle]
                avrg_vel = np.nanmean(cardiac_cycle_array)
                max_vel = np.amax(cardiac_cycle_array)
                self.mean_velocity_cardiac_cycle.append(avrg_vel)
                self.peak_velocity_cardiac_cycle.append(max_vel)
                # Q = 60 * A * v_mean / Q_expected =~ 600 ml/min
                #num_pixels = np.sum(self.mask[..., cardiac_cycle])
                num_pixels = np.count_nonzero(~np.isnan(cardiac_cycle_array))
                # if avrg_vel > 0: Q ; else: -Q?
                Q = 60 * num_pixels * 0.1 * self.pixel_spacing[0] * \
                    0.1 * self.pixel_spacing[1] * avrg_vel
                # (mm * 0.1 * mm * 0.1) = cm2 ; (cm2 * cm/s * 60s) = cm3/min = mm3/min
                self.RBF.append(Q)
            self.mean_velocity = np.mean(self.mean_velocity_cardiac_cycle)
            self.peak_velocity = np.amax(self.peak_velocity_cardiac_cycle)
            self.mean_RBF = np.mean(self.RBF)
            # Convert any nan values back to 0
            self.velocity_array = np.nan_to_num(self.velocity_array)
            self.mask = np.nan_to_num(self.mask)
        else:
            raise ValueError('The input velocity_array should be 3-dimensional.')

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


def phase_to_velocity(phase_array, velocity_encoding):
    """
    Calculate the velocity array from the given input phase image and velocity.

    Parameters
    ----------
    phase_array: np.ndarray
        A 3D array containing the phase images of the phase contrast sequence.
    velocity_encoding : float
        The value of the velocity encoding in cm/s.

    Returns
    -------
    velocity_array: np.ndarray
    """
    # (v = phase_delta / np.pi ) * velocity_encoding
    # v_expected =~ 20cm/s
    return convert_to_pi_range(phase_array) * velocity_encoding