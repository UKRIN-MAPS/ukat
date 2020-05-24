"""Data fetcher module. Do not move this file from `data` folder."""

import os
import json
import numpy as np
import nibabel as nib

DIR_DATA = os.path.dirname(os.path.realpath(__file__))

def r2star_ge():
    """Fetches r2star/ge dataset

    Several sentences providing an extended description. Refer to variables
    using back-ticks, e.g. `var`.

    Returns
    -------
    numpy.ndarray
        image data
    numpy.ndarray
        affine matrix for image data
    numpy.ndarray
        list of echo times, in seconds

    """

    # Initialise hard-coded list of file names that are the expected files
    # in this test dataset. If the actual files in the directory don't match
    # this list this means that the test dataset has been corrupted.
    # Note that these file names are sorted alphabetically and not sorted by
    # increasing echo time. The sort by echo time will be done later below.
    expected_files = ['00016__InPhase_Cor_R2_Mapping_BH_+_phase_e1.json',
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

    # Get filenames ensuring alphabetical order to match expected_files
    dir_r2star_ge = os.path.join(DIR_DATA, "r2star", "ge")
    files = sorted(os.listdir(dir_r2star_ge))

    # Check filenames match the expected dataset
    not_match_msg = (f"Expected files in {dir_r2star_ge}:\n{expected_files}\n"
                     f"This doesn't match the files that were found:\n{files}")
    assert (files == expected_files), not_match_msg

    # Load magnitude data and corresponding echo times (in the orig)
    image = []
    echo_list = []
    for file in files:
        filepath = os.path.join(dir_r2star_ge, file)

        if file.endswith(".nii.gz"):

            # Load NIfTI and only save the magnitude data (index 0)
            data = nib.load(filepath)
            image.append(data.get_fdata()[..., 0])

        elif file.endswith(".json"):

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
