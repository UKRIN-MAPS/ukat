"""Data fetcher module. Do not move this file from `data` folder.

The data fetcher function names match the directory structure of the test data.
For example, to load data in the `data/dwi/ge` directory, there should be a
fetcher function named `dwi_ge`.

"""

import os
import json
import numpy as np
import nibabel as nib
#from dipy.io import read_bvals_bvecsgit

DIR_DATA = os.path.dirname(os.path.realpath(__file__))


def get_filepaths(directory, expected_filenames):
    """Get filepaths in directory; check their names match expected_filenames

    Parameters
    ----------
    directory : str
        full path to directory
    expected_filenames : list
        list of strings with expected file names

    Returns
    -------
    list
        list of strings with full paths to files (sorted alphabetically)

    """
    # Ensure expected_filenames is sorted alphabetically
    expected_filenames = sorted(expected_filenames)

    # Get filenames in directory ensuring alphabetical order
    filenames = sorted(os.listdir(directory))

    # Check filenames match expected_filenames
    not_match_msg = (f"Expected files in {directory}:\n{expected_filenames}\n"
                     f"This doesn't match the found files:\n{filenames}")
    assert (filenames == expected_filenames), not_match_msg

    # Make list of file paths
    filepaths = []
    for filename in filenames:
        filepaths.append(os.path.join(directory, filename))

    return filepaths


def dwi_ge():
    """Fetches dwi/ge dataset

    Returns
    -------
    numpy.ndarray
        image data
    numpy.ndarray
        affine matrix for image data
    numpy.ndarray
        array of b-values
    numpy.ndarray
        array of b-vectors

    """

    # Initialise hard-coded list of file names that are the expected files
    # in this test dataset. If the actual files in the directory don't match
    # this list this means that the test dataset has been corrupted.
    expected_filenames = ['00014__Cor_DWI_RT.bval',
                          '00014__Cor_DWI_RT.bvec',
                          '00014__Cor_DWI_RT.json',
                          '00014__Cor_DWI_RT.nii.gz']

    # Initialise path to dwi/ge
    dir_dwi_ge = os.path.join(DIR_DATA, "dwi", "ge")

    # Get filepaths in directory and check their names match expected_filenames
    filepaths = get_filepaths(dir_dwi_ge, expected_filenames)

    # Read bvals, bvecs, and DWI data into numpy arrays
    bval_path = filepaths[0]
    bvec_path = filepaths[1]
    nii_path = filepaths[3]

    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    nii = nib.load(nii_path)

    data = nii.get_fdata()
    affine = nii.affine

    return data, affine, bvals, bvecs


def r2star_ge():
    """Fetches r2star/ge dataset

    Returns
    -------
    numpy.ndarray
        image data
    numpy.ndarray
        affine matrix for image data
    numpy.ndarray
        array of echo times, in seconds

    """

    # Initialise hard-coded list of file names that are the expected files
    # in this test dataset. If the actual files in the directory don't match
    # this list this means that the test dataset has been corrupted.
    # Note that these file names are sorted alphabetically and not sorted by
    # increasing echo time. The sort by echo time will be done later below.
    expected_filenames = ['00016__InPhase_Cor_R2_Mapping_BH_+_phase_e1.json',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e1.nii.gz',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e10.json',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e10.nii.gz',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e11.json',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e11.nii.gz',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e12.json',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e12.nii.gz',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e2.json',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e2.nii.gz',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e3.json',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e3.nii.gz',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e4.json',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e4.nii.gz',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e5.json',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e5.nii.gz',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e6.json',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e6.nii.gz',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e7.json',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e7.nii.gz',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e8.json',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e8.nii.gz',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e9.json',
                          '00016__InPhase_Cor_R2_Mapping_BH_+_phase_e9.nii.gz']

    # Initialise path to r2star/ge
    dir_r2star_ge = os.path.join(DIR_DATA, "r2star", "ge")

    # Get filepaths in directory and check their names match expected_filenames
    filepaths = get_filepaths(dir_r2star_ge, expected_filenames)

    # Load magnitude data and corresponding echo times (in the orig)
    image = []
    echo_list = []
    for filepath in filepaths:

        if filepath.endswith(".nii.gz"):

            # Load NIfTI and only save the magnitude data (index 0)
            data = nib.load(filepath)
            image.append(data.get_fdata()[..., 0])

        elif filepath.endswith(".json"):

            # Retrieve list of echo times in the original order
            with open(filepath, 'r') as json_file:
                hdr = json.load(json_file)
            echo_list.append(hdr['EchoTime'])

    # Move echo dimension to 4th dimension
    image = np.moveaxis(np.array(image), 0, -1)
    echo_list = np.array(echo_list)

    # Sort by increasing echo time
    sort_idxs = np.argsort(echo_list)
    echo_list = echo_list[sort_idxs]
    image = image[:, :, :, sort_idxs]

    return image, data.affine, echo_list



def r2star_siemens():
    """Fetches r2star/siemens dataset

    Returns
    -------
    numpy.ndarray
        image data
    numpy.ndarray
        affine matrix for image data
    numpy.ndarray
        array of echo times, in seconds

    """

    # Initialise hard-coded list of file names that are the expected files
    # in this test dataset. If the actual files in the directory don't match
    # this list this means that the test dataset has been corrupted.
    # Note that these file names are sorted alphabetically and not sorted by
    # increasing echo time. The sort by echo time will be done later below.
    expected_filenames = ['00025__bh3x_r2star_inphase_volume_e1.json',
                          '00025__bh3x_r2star_inphase_volume_e1.nii.gz',
                          '00025__bh3x_r2star_inphase_volume_e10.json',
                          '00025__bh3x_r2star_inphase_volume_e10.nii.gz',
                          '00025__bh3x_r2star_inphase_volume_e11.json',
                          '00025__bh3x_r2star_inphase_volume_e11.nii.gz',
                          '00025__bh3x_r2star_inphase_volume_e12.json',
                          '00025__bh3x_r2star_inphase_volume_e12.nii.gz',
                          '00025__bh3x_r2star_inphase_volume_e2.json',
                          '00025__bh3x_r2star_inphase_volume_e2.nii.gz',
                          '00025__bh3x_r2star_inphase_volume_e3.json',
                          '00025__bh3x_r2star_inphase_volume_e3.nii.gz',
                          '00025__bh3x_r2star_inphase_volume_e4.json',
                          '00025__bh3x_r2star_inphase_volume_e4.nii.gz',
                          '00025__bh3x_r2star_inphase_volume_e5.json',
                          '00025__bh3x_r2star_inphase_volume_e5.nii.gz',
                          '00025__bh3x_r2star_inphase_volume_e6.json',
                          '00025__bh3x_r2star_inphase_volume_e6.nii.gz',
                          '00025__bh3x_r2star_inphase_volume_e7.json',
                          '00025__bh3x_r2star_inphase_volume_e7.nii.gz',
                          '00025__bh3x_r2star_inphase_volume_e8.json',
                          '00025__bh3x_r2star_inphase_volume_e8.nii.gz',
                          '00025__bh3x_r2star_inphase_volume_e9.json',
                          '00025__bh3x_r2star_inphase_volume_e9.nii.gz']

    # Initialise path to r2star/siemens
    dir_r2star_siemens = os.path.join(DIR_DATA, "r2star", "siemens")

    # Get filepaths in directory and check their names match expected_filenames
    filepaths = get_filepaths(dir_r2star_siemens, expected_filenames)

    # Load magnitude data and corresponding echo times (in the orig)
    image = []
    echo_list = []
    for filepath in filepaths:

        if filepath.endswith(".nii.gz"):

            # Load NIfTI and only save the magnitude data (index 0)
            data = nib.load(filepath)
            image.append(data.get_fdata())

        elif filepath.endswith(".json"):

            # Retrieve list of echo times in the original order
            with open(filepath, 'r') as json_file:
                hdr = json.load(json_file)
            echo_list.append(hdr['EchoTime'])

    # Move echo dimension to 4th dimension
    image = np.moveaxis(np.array(image), 0, -1)
    echo_list = np.array(echo_list)

    # Sort by increasing echo time
    sort_idxs = np.argsort(echo_list)
    echo_list = echo_list[sort_idxs]
    image = image[:, :, :, sort_idxs]

    return image, data.affine, echo_list


def r2star_philips():
    """Fetches r2star/philips dataset

    Returns
    -------
    numpy.ndarray
        image data
    numpy.ndarray
        affine matrix for image data
    numpy.ndarray
        array of echo times, in seconds

    """

    # Initialise hard-coded list of file names that are the expected files
    # in this test dataset. If the actual files in the directory don't match
    # this list this means that the test dataset has been corrupted.
    # Note that these file names are sorted alphabetically and not sorted by
    # increasing echo time. The sort by echo time will be done later below.

    expected_filenames = ['01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e1.json',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e1.nii.gz',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e10.json',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e10.nii.gz',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e11.json',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e11.nii.gz',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e12.json',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e12.nii.gz',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e13.json',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e13.nii.gz',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e3.json',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e3.nii.gz',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e4.json',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e4.nii.gz',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e5.json',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e5.nii.gz',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e6.json',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e6.nii.gz',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e7.json',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e7.nii.gz',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e8.json',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e8.nii.gz',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e9.json',
                          '01401__Kidney_T2star_m-FFE_3x3x5_SPIR_volume_inphase_e9.nii.gz']

    # Initialise path to r2star/philips
    dir_r2star_philips = os.path.join(DIR_DATA, "r2star", "philips")

    # Get filepaths in directory and check their names match expected_filenames
    filepaths = get_filepaths(dir_r2star_philips, expected_filenames)

    # Load magnitude data and corresponding echo times (in the orig)
    image = []
    echo_list = []
    for filepath in filepaths:

        if filepath.endswith(".nii.gz"):

            # Load NIfTI and only save the magnitude data (index 0)
            data = nib.load(filepath)

            image.append(data.get_fdata())

        elif filepath.endswith(".json"):

            # Retrieve list of echo times in the original order
            with open(filepath, 'r') as json_file:

                hdr = json.load(json_file)


            echo_list.append(hdr["EchoTime"])

    # Move echo dimension to 4th dimension
    image = np.moveaxis(np.array(image), 0, -1)
    echo_list = np.array(echo_list)


    # Sort by increasing echo time
    sort_idxs = np.argsort(echo_list)
    echo_list = echo_list[sort_idxs]
    image = image[:, :, :, sort_idxs]

    return image, data.affine, echo_list
