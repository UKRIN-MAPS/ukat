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
    phase_array : np.ndarray
        The input pixel array provided to the class initialisation.
    velocity_array : np.ndarray
        The phase_array converted to velocity based in velocity_encoding.
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

    def __init__(self, pixel_array, velocity_encoding, affine, mask=None):
        """Initialise a PhaseContrast class instance.

        Parameters
        ----------
        pixel_array : np.ndarray
            A 3D array containing the phase images of the phase contrast
            sequence, i.e. the dimensions of the array are [x, y, c], where c
            corresponds to the cardica cycle.
        velocity_encoding : float
            The value of the velocity encoding in cm/s.
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

        self.phase_array = pixel_array
        self.velocity_encoding = velocity_encoding
        self.shape = pixel_array.shape
        self.affine = affine
        self.pixel_spacing = [np.linalg.norm(self.affine[:3, 1]),
                              np.linalg.norm(self.affine[:3, 0])]
        # Generate a mask if there isn't one specified
        if mask is None:
            self.mask = np.ones(self.shape, dtype=bool)
        else:
            self.mask = mask
        self.velocity_array = np.zeros(self.shape)
        self.mean_velocity_cardiac_cycle = []
        self.peak_velocity_cardiac_cycle = []
        self.RBF = []
        self.mean_velocity = 0
        self.peak_velocity = 0
        self.mean_RBF = 0

        if len(self.shape) == 3:
            # v = (phase / np.pi) * v_enc
            self.velocity_array = convert_to_pi_range(self.phase_array) * \
                                    self.velocity_encoding * self.mask
            
            for cardiac_cycle in range(self.shape[-1]):
                cardiac_cycle_array = self.velocity_array[..., cardiac_cycle]
                avrg_vel = np.nanmean(cardiac_cycle_array)
                max_vel = np.amax(cardiac_cycle_array)
                self.mean_velocity_cardiac_cycle.append(avrg_vel)
                self.peak_velocity_cardiac_cycle.append(max_vel)
                # Q = 60 * A * v_mean
                flow = 60 * np.count_nonzero(cardiac_cycle_array) * \
                       self.pixel_spacing[0] * 10 * self.pixel_spacing[1] * \
                       10 * avrg_vel
                # 10 * cm/s = mm/s ; 0.001 mm3 = 1 cm3 = 1 ml ; 60 sec = 1 min
                self.RBF.append(flow)
            
            self.mean_velocity = np.mean(self.mean_velocity_cardiac_cycle)
            self.peak_velocity = np.amax(self.peak_velocity_cardiac_cycle)
            self.mean_RBF = np.mean(self.RBF)

        else:
            raise ValueError('The input pixel_array should be 3-dimensional.')

    def to_nifti(self, output_directory=os.getcwd(), base_file_name='Output',
                 maps='all'):
        """Exports the input pixel array and/or the velocity array to NIFTI.

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
            maps = ["phase_array", "mask", "velocity_array"]
        if isinstance(maps, list):
            for result in maps:
                if result == 'phase_array' or result == 'pixel_array':
                    phase_nifti = nib.Nifti1Image(self.phase_array,
                                               affine=self.affine)
                    nib.save(phase_nifti, base_path + '_phase_array.nii.gz')
                elif result == 'mask':
                    mask_nifti = nib.Nifti1Image(self.mask.astype(int),
                                                 affine=self.affine)
                    nib.save(mask_nifti, base_path + '_mask.nii.gz')
                elif result == 'velocity_array' or result == 'velocity array':
                    velocity_nifti = nib.Nifti1Image(self.velocity_array,
                                                 affine=self.affine)
                    nib.save(velocity_nifti, base_path + '_velocity_array.nii.gz')
        else:
            raise ValueError('No NIFTI file saved. The variable "maps" '
                             'should be "all" or a list of maps from '
                             '"["phase_array", "mask", "velocity_array"]".')

        return
