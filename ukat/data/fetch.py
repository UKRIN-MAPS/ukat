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

    # Initialise path to dwi/ge
    dir_dwi_ge = os.path.join(DIR_DATA, "dwi", "ge")

    # Get filepaths in directory
    filepaths = sorted([os.path.join(dir_dwi_ge, f) for f in
                        os.listdir(dir_dwi_ge) if not f.endswith('.py')])

    # Read bvals, bvecs, and DWI data into numpy arrays
    bval_path = [f for f in filepaths if f.endswith('.bval')][0]
    bvec_path = [f for f in filepaths if f.endswith('.bvec')][0]
    nii_path = [f for f in filepaths if f.endswith('.nii.gz')][0]

    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    nii = nib.load(nii_path)

    data = nii.get_fdata()
    affine = nii.affine

    return data, affine, bvals, bvecs


def dwi_philips():
    """Fetches dwi/philips dataset

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

    # Initialise path to dwi/philips
    dir_dwi_philips = os.path.join(DIR_DATA, "dwi", "philips")

    # Get filepaths in directory
    filepaths = sorted([os.path.join(dir_dwi_philips, f) for f in
                        os.listdir(dir_dwi_philips) if not f.endswith('.py')])

    # Read bvals, bvecs, and DWI data into numpy arrays
    bval_path = [f for f in filepaths if f.endswith('.bval')][0]
    bvec_path = [f for f in filepaths if f.endswith('.bvec')][0]
    nii_path = [f for f in filepaths if f.endswith('.nii.gz')][0]

    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    nii = nib.load(nii_path)

    data = nii.get_fdata()
    affine = nii.affine

    return data, affine, bvals, bvecs


def dwi_siemens():
    """Fetches dwi/siemens dataset

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

    # Initialise path to dwi/siemens
    dir_dwi_siemens = os.path.join(DIR_DATA, "dwi", "siemens")

    # Get filepaths in directory
    filepaths = sorted([os.path.join(dir_dwi_siemens, f) for f in
                        os.listdir(dir_dwi_siemens) if not f.endswith('.py')])

    # Read bvals, bvecs, and DWI data into numpy arrays
    bval_path = [f for f in filepaths if f.endswith('.bval')][0]
    bvec_path = [f for f in filepaths if f.endswith('.bvec')][0]
    nii_path = [f for f in filepaths if f.endswith('.nii.gz')][0]

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

    # Initialise path to r2star/ge
    dir_r2star_ge = os.path.join(DIR_DATA, "r2star", "ge")

    # Get filepaths in directory
    filepaths = sorted([os.path.join(dir_r2star_ge, f) for f in
                        os.listdir(dir_r2star_ge) if not f.endswith('.py')])

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

    # Initialise path to r2star/siemens
    dir_r2star_siemens = os.path.join(DIR_DATA, "r2star", "siemens")

    # Get filepaths in directory
    filepaths = sorted([os.path.join(dir_r2star_siemens, f) for f in
                        os.listdir(dir_r2star_siemens)
                        if not f.endswith('.py')])

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

    # Initialise path to r2star/philips
    dir_r2star_philips = os.path.join(DIR_DATA, "r2star", "philips")

    # Get filepaths in directory
    filepaths = sorted([os.path.join(dir_r2star_philips, f) for f in
                        os.listdir(dir_r2star_philips)
                        if not f.endswith('.py')])

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

    # Initialise path to b0/ge
    dir_b0_ge = os.path.join(DIR_DATA, "b0", "ge")

    # Get filepaths in directory
    filepaths = sorted([os.path.join(dir_b0_ge, f) for f in
                        os.listdir(dir_b0_ge) if not f.endswith('.py')])

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


def _load_b0_siemens_philips(filepaths):
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
    See outputs of _load_b0_siemens_philips
    """

    POSSIBLE_DATASET_IDS = [1, 2]

    if (dataset_id != 1) and (dataset_id != 2):
        error_msg = f"`dataset_id` must be one of {POSSIBLE_DATASET_IDS}"
        raise ValueError(error_msg)

    # Initialise path to b0/siemens_{dataset_id}
    dir_b0_siemens = os.path.join(DIR_DATA, "b0", "siemens" + f"_{dataset_id}")

    # Get filepaths in directory
    filepaths = sorted([os.path.join(dir_b0_siemens, f) for f in
                        os.listdir(dir_b0_siemens) if not f.endswith('.py')])

    # Load data
    magnitude, phase, affine, echo_times = _load_b0_siemens_philips(filepaths)

    return magnitude, phase, affine, echo_times


def b0_philips(dataset_id):
    """Fetches b0/philips_{dataset_id} dataset
    dataset_id : int
        Number of the dataset to load:
        - dataset_id = 1 to load "b0\philips_1"
        - dataset_id = 2 to load "b0\philips_2"
    Returns
    -------
    See outputs of _load_b0_siemens_philips
    """

    POSSIBLE_DATASET_IDS = [1, 2]

    if (dataset_id != 1) and (dataset_id != 2):
        error_msg = f"`dataset_id` must be one of {POSSIBLE_DATASET_IDS}"
        raise ValueError(error_msg)

    # Initialise path to b0/philips_{dataset_id}
    dir_b0_philips = os.path.join(DIR_DATA, "b0", "philips" + f"_{dataset_id}")

    # Get filepaths in directory
    filepaths = sorted([os.path.join(dir_b0_philips, f) for f in
                        os.listdir(dir_b0_philips) if not f.endswith('.py')])

    # Load data
    if dataset_id == 2:
        magnitude, phase, affine, echo_times = _load_b0_siemens_philips(
            filepaths)
    elif dataset_id == 1:
        error_msg = ("Functionality to read datasets where phase data was not "
                     "saved for both echoes separately is not implemented")
        raise ValueError(error_msg)

    return magnitude, phase, affine, echo_times


def t1_philips(dataset_id):
    """Fetches t1/philips_{dataset_id} dataset
    dataset_id : int
            Number of the dataset to load:
            - dataset_id = 1 to load "t1\philips_1"
            - dataset_id = 2 to load "t1\philips_2"
    Returns
    -------
    numpy.ndarray
        image data
    numpy.ndarray
        affine matrix for image data
    numpy.ndarray
        array of inversion times, in seconds
    float
        temporal slice spacing of image, in seconds
    """

    POSSIBLE_DATASET_IDS = [1, 2]

    if (dataset_id != 1) and (dataset_id != 2):
        error_msg = f"`dataset_id` must be one of {POSSIBLE_DATASET_IDS}"
        raise ValueError(error_msg)

    # Initialise path to t1/philips_{dataset_id}
    dir_t1_philips = os.path.join(DIR_DATA, 't1', 'philips' + f'_{dataset_id}')

    # Get filepaths in directory
    filepaths = sorted([os.path.join(dir_t1_philips, f) for f in
                        os.listdir(dir_t1_philips) if not f.endswith('.py')])

    # See README.md in ukat\data\t1 for information about the acquisition.
    if dataset_id == 1:
        # Load magnitude data and corresponding inversion times (in the orig)
        image = []
        inversion_list = []
        for filepath in filepaths:

            if filepath.endswith(".nii.gz"):

                # Load NIfTI and only save the magnitude data (index 0)
                data = nib.load(filepath)
                image.append(data.get_fdata())

            elif filepath.endswith(".json"):

                # Retrieve list of echo times in the original order
                with open(filepath, 'r') as json_file:
                    hdr = json.load(json_file)
                inversion_list.append(hdr["InversionTime"])

        # Move inversion dimension to 4th dimension
        image = np.moveaxis(np.array(image), 0, -1)
        inversion_list = np.array(inversion_list)

        # Sort by increasing inversion time
        sort_idxs = np.argsort(inversion_list)
        inversion_list = inversion_list[sort_idxs]
        tss = 0
        magnitude = image[:, :, :, sort_idxs]
        phase = np.zeros(image.shape)
        affine = data.affine

    elif dataset_id == 2:
        magnitude_path = [f for f in filepaths if ("__ph" not in f)
                          and f.endswith('.nii.gz')][0]
        magnitude_img = nib.load(magnitude_path)
        magnitude = magnitude_img.get_fdata()
        phase_path = [f for f in filepaths if f.endswith('__ph.nii.gz')][0]
        phase_img = nib.load(phase_path)
        phase = phase_img.get_fdata()
        inversion_list = np.arange(0.1, 1.801, 0.1)
        tss = 0.0537
        affine = magnitude_img.affine

    return magnitude, phase, affine, inversion_list, tss

def t2_philips(dataset_id):
    """Fetches t2/philips_{dataset_id} dataset
    dataset_id : int
            Number of the dataset to load:
            - dataset_id = 1 to load "t2\philips_1"
    Returns
    -------
    numpy.ndarray
        image data
    numpy.ndarray
        affine matrix for image data
    numpy.ndarray
        array of echo times, in seconds
    """

    POSSIBLE_DATASET_IDS = [1]

    if dataset_id != 1:
        error_msg = f"`dataset_id` must be one of {POSSIBLE_DATASET_IDS}"
        raise ValueError(error_msg)

    # Initialise path to t1/philips_{dataset_id}
    dir_t2_philips = os.path.join(DIR_DATA, 't2', 'philips' + f'_{dataset_id}')

    # Get filepaths in directory
    filepaths = sorted([os.path.join(dir_t2_philips, f) for f in
                        os.listdir(dir_t2_philips) if not f.endswith('.py')])

    # See README.md in ukat\data\t2 for information about the acquisition.
    if dataset_id == 1:
        # Load magnitude data and corresponding echo times (in the orig)
        magnitude = []
        echo_list = []
        for filepath in filepaths:

            if filepath.endswith(".nii.gz"):

                # Load NIfTI
                data = nib.load(filepath)
                magnitude.append(data.get_fdata())

            elif filepath.endswith(".json"):

                # Retrieve list of echo times in the original order
                with open(filepath, 'r') as json_file:
                    hdr = json.load(json_file)
                echo_list.append(hdr["EchoTime"])

        # Move echo dimension to 4th dimension
        magnitude = np.moveaxis(np.array(magnitude), 0, -1)
        echo_list = np.array(echo_list)

        # Sort by increasing echo time
        sort_idxs = np.argsort(echo_list)
        echo_list = echo_list[sort_idxs]
        magnitude = magnitude[:, :, :, sort_idxs]
        affine = data.affine

    return magnitude, affine, echo_list
