"""Data fetcher module. Do not move this file from `data` folder.

The data fetcher function names match the directory structure of the test data.
For example, to load data in the `data/dwi/ge` directory, there should be a
fetcher function named `dwi_ge`.

"""

import os
import json
import numpy as np
import nibabel as nib
from dipy.io import read_bvals_bvecs

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

    # Ignore "__init__.py" files
    if "__init__.py" in filenames:
        filenames.remove("__init__.py")

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
    image = image[..., sort_idxs]

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


def b0_ge():
    """Fetches b0/ge dataset

    Returns
    -------
    numpy.ndarray
        image data - Magnitude
    numpy.ndarray
        image data - Phase
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
    expected_filenames = ['00009__3D_B0_map_VOL_e1.json',
                          '00009__3D_B0_map_VOL_e1.nii.gz',
                          '00009__3D_B0_map_VOL_e2.json',
                          '00009__3D_B0_map_VOL_e2.nii.gz']

    # Initialise path to b0/ge
    dir_b0_ge = os.path.join(DIR_DATA, "b0", "ge")

    # Get filepaths in directory and check their names match expected_filenames
    filepaths = get_filepaths(dir_b0_ge, expected_filenames)

    # Load magnitude, real and imaginary data and corresponding echo times
    magnitude = []
    real = []
    imaginary = []
    echo_list = []
    for filepath in filepaths:

        if filepath.endswith(".nii.gz"):

            # Load NIfTI and save the magnitude data (index 0)
            data = nib.load(filepath)
            magnitude.append(data.get_fdata()[..., 0])
            # Save the real data (index 1) - I STILL NEED TO CONFIRM!
            real.append(data.get_fdata()[..., 1])
            # Save the imaginary data (index 2) - I STILL NEED TO CONFIRM!
            imaginary.append(data.get_fdata()[..., 2])

        elif filepath.endswith(".json"):

            # Retrieve list of echo times in the original order
            with open(filepath, 'r') as json_file:
                hdr = json.load(json_file)
            echo_list.append(hdr['EchoTime'])

    # Move echo dimension to 4th dimension
    magnitude = np.moveaxis(np.array(magnitude), 0, -1)
    real = np.moveaxis(np.array(real), 0, -1)
    imaginary = np.moveaxis(np.array(imaginary), 0, -1)

    # Calculate Phase image => tan-1(Im/Re)
    # np.negative is used to change the sign - as discussed with Andy Priest
    phase = np.negative(np.arctan2(imaginary, real))

    echo_list = np.array(echo_list)

    # Sort by increasing echo time
    sort_idxs = np.argsort(echo_list)
    echo_list = echo_list[sort_idxs]
    magnitude = magnitude[..., sort_idxs]
    phase = phase[..., sort_idxs]

    return magnitude, phase, data.affine, echo_list


def _load_b0_siemens(filepaths):
    """General function to retrieve siemens b0 data from list of filepaths

    Returns
    -------
    numpy.ndarray
        image data - Magnitude
    numpy.ndarray
        image data - Phase
    numpy.ndarray
        affine matrix for image data
    numpy.ndarray
        array of echo times, in seconds

    """
    # Load magnitude, real and imaginary data and corresponding echo times
    data = []
    affines = []
    image_types = []
    echo_times = []

    for filepath in filepaths:

        if filepath.endswith(".nii.gz"):

            # Load data in NIfTI files
            nii = nib.load(filepath)
            data.append(nii.get_fdata())
            affines.append(nii.affine)

            # Load necessary information from corresponding .json files
            json_path = filepath.replace(".nii.gz", ".json")
            with open(json_path, 'r') as json_file:
                hdr = json.load(json_file)
                image_types.append(hdr['ImageType'])
                echo_times.append(hdr['EchoTime'])

    # Sort by increasing echo time
    sort_idxs = np.argsort(echo_times)
    data = np.array([data[i] for i in sort_idxs])
    echo_times = np.array([echo_times[i] for i in sort_idxs])
    image_types = [image_types[i] for i in sort_idxs]

    # Move measurements (time) dimension to 4th dimension
    data = np.moveaxis(data, 0, -1)

    # Separate magnitude and phase images
    magnitude_idxs = ["M" in i for i in image_types]
    phase_idxs = ["P" in i for i in image_types]

    magnitude = data[..., magnitude_idxs]
    phase = data[..., phase_idxs]

    echo_times_magnitude = echo_times[magnitude_idxs]
    echo_times_phase = echo_times[phase_idxs]

    # Assign unique echo times for output
    echo_times_are_equal = (echo_times_magnitude == echo_times_phase).all()
    if echo_times_are_equal:
        echo_times = echo_times_magnitude
    else:
        raise ValueError("Magnitude and phase echo times must be equal")

    # If all affines are equal, initialise the affine for output
    affines_are_equal = (np.array([i == affines[0] for i in affines])).all()
    if affines_are_equal:
        affine = affines[0]
    else:
        raise ValueError("Affine matrices of input data are not all equal")

    return magnitude, phase, affine, echo_times


def b0_siemens(dataset_id):
    """Fetches b0/siemens_{dataset_id} dataset

    dataset_id : int
        Number of the dataset to load:
        - dataset_id = 1 to load "b0\siemens_1"
        - dataset_id = 2 to load "b0\siemens_2"

    Returns
    -------
    See outputs of _load_b0_siemens

    """

    POSSIBLE_DATASET_IDS = [1, 2]

    # Initialise hard-coded list of file names that are the expected files
    # in the test dataset. If the actual files in the directory don't match
    # this list this means that the test dataset has been corrupted.
    # Note these file names are sorted alphabetically and may not be sorted
    # by increasing echo time. The sort by echo time will be done later below.
    if dataset_id == 1:
        expected_filenames = ['00010__bh_b0map_3D_default_e1.json',
                              '00010__bh_b0map_3D_default_e1.nii.gz',
                              '00010__bh_b0map_3D_default_e2.json',
                              '00010__bh_b0map_3D_default_e2.nii.gz',
                              '00011__bh_b0map_3D_default_e1.json',
                              '00011__bh_b0map_3D_default_e1.nii.gz',
                              '00011__bh_b0map_3D_default_e2.json',
                              '00011__bh_b0map_3D_default_e2.nii.gz']
    elif dataset_id == 2:
        expected_filenames = ['00044__bh_b0map_fa3_default_e1.json',
                              '00044__bh_b0map_fa3_default_e1.nii.gz',
                              '00044__bh_b0map_fa3_default_e2.json',
                              '00044__bh_b0map_fa3_default_e2.nii.gz',
                              '00045__bh_b0map_fa3_default_e1.json',
                              '00045__bh_b0map_fa3_default_e1.nii.gz',
                              '00045__bh_b0map_fa3_default_e2.json',
                              '00045__bh_b0map_fa3_default_e2.nii.gz']
    else:
        error_msg = f"`dataset_id` must be one of {POSSIBLE_DATASET_IDS}"
        raise ValueError(error_msg)

    # Initialise path to b0/siemens_{dataset_id}
    dir_b0_siemens = os.path.join(DIR_DATA, "b0", "siemens" + f"_{dataset_id}")

    # Get filepaths in directory and check their names match expected_filenames
    filepaths = get_filepaths(dir_b0_siemens, expected_filenames)

    # Load data
    magnitude, phase, affine, echo_times = _load_b0_siemens(filepaths)

    return magnitude, phase, affine, echo_times
