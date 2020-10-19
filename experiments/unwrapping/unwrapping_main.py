# Note: GE scanner-calculated b0 maps are multiplied by 10

import os

from unwrapping import CompareUnwrapping

# CONSTANTS
DIR_ROOT = os.getcwd()
DIR_DATA = os.path.join(DIR_ROOT, "data")

# DATASETS correspond to the name of each subdir with each dataset
DATASETS = ['ge_01', 'philips_01', 'philips_02', 'philips_03',
            'siemens_01', 'philips_04', 'philips_05', 'philips_06',
            'philips_07', 'philips_08', 'philips_09', 'philips_10',
            'philips_11', 'philips_12', 'philips_13', 'philips_14',
            'philips_15', 'philips_16', 'philips_17', 'siemens_02',
            'siemens_03', 'siemens_04', 'siemens_05']

# Don't change slices to show as some datasets only have ROIs drawn for this
# particular slice
SLICES_TO_SHOW = [8, 4, 4, 4,
                  4, 4, 4, 4,
                  4, 4, 4, 4,
                  4, 4, 4, 4,
                  4, 4, 4, 4,
                  4, 5, 5]

FILENAMES_ALL = []
FILENAMES_ALL.append(['00720__Magnitude_ims__B0_map_dual_echo_e1.nii.gz',  # ge_01
                      '00721__Phase_ims__B0_map_dual_echo_e1.nii.gz',
                      '00720__Magnitude_ims__B0_map_dual_echo_e2.nii.gz',
                      '00721__Phase_ims__B0_map_dual_echo_e2.nii.gz',
                      '00720__Magnitude_ims__B0_map_dual_echo_e1_roi-R.nii.gz',
                      '00720__Magnitude_ims__B0_map_dual_echo_e1_roi-L.nii.gz',
                      '00730__B0_off-resonance_maps_(Hz_x_10).nii.gz'])
FILENAMES_ALL.append(['01001__B0_map_expiration_volume_2DMS_product_auto_e1.nii.gz',  # philips_01
                      '01001__B0_map_expiration_volume_2DMS_product_auto_e1_ph.nii.gz',
                      '01001__B0_map_expiration_volume_2DMS_product_auto_e2.nii.gz',
                      '01001__B0_map_expiration_volume_2DMS_product_auto_e2_ph.nii.gz',
                      '01001__B0_map_expiration_volume_2DMS_product_auto_e1_roi-R.nii.gz',
                      '01001__B0_map_expiration_volume_2DMS_product_auto_e1_roi-L.nii.gz',
                      '00501__B0_map_expiration_volume_2DMS_product_auto_e1a.nii.gz'])
FILENAMES_ALL.append(['01101__B0_map_expiration_volume_2DMS_product_auto_e1.nii.gz',  # philips_02
                      '01101__B0_map_expiration_volume_2DMS_product_auto_e1_ph.nii.gz',
                      '01101__B0_map_expiration_volume_2DMS_product_auto_e2.nii.gz',
                      '01101__B0_map_expiration_volume_2DMS_product_auto_e2_ph.nii.gz',
                      '01101__B0_map_expiration_volume_2DMS_product_auto_e1_roi-R.nii.gz',
                      '01101__B0_map_expiration_volume_2DMS_product_auto_e1_roi-L.nii.gz',
                      '00701__B0_map_expiration_volume_2DMS_product_auto_e1a.nii.gz'])
FILENAMES_ALL.append(['01301__B0_map_expiration_volume_2DMS_product_auto_e1.nii.gz',  # philips_03
                      '01301__B0_map_expiration_volume_2DMS_product_auto_e1_ph.nii.gz',
                      '01301__B0_map_expiration_volume_2DMS_product_auto_e2.nii.gz',
                      '01301__B0_map_expiration_volume_2DMS_product_auto_e2_ph.nii.gz',
                      '01301__B0_map_expiration_volume_2DMS_product_auto_e1_roi-R.nii.gz',
                      '01301__B0_map_expiration_volume_2DMS_product_auto_e1_roi-L.nii.gz',
                      '01201__B0_map_expiration_volume_2DMS_product_auto_e1a.nii.gz'])
FILENAMES_ALL.append(['0044_bh_b0map_fa3_default_bh_b0map_fa3_default_e1.nii.gz',  # siemens_01
                      '0045_bh_b0map_fa3_default_bh_b0map_fa3_default_e1_ph.nii.gz',
                      '0044_bh_b0map_fa3_default_bh_b0map_fa3_default_e2.nii.gz',
                      '0045_bh_b0map_fa3_default_bh_b0map_fa3_default_e2_ph.nii.gz',
                      '0044_bh_b0map_fa3_default_bh_b0map_fa3_default_e1_roi-R.nii.gz',
                      '0044_bh_b0map_fa3_default_bh_b0map_fa3_default_e1_roi-L.nii.gz',
                      ''])  # scanner-generated B0 map not available on Siemens
FILENAMES_ALL.append(['00501__B0_map_expiration_volume_2DMS_delTE2.3_e1.nii.gz',  # philips_04
                      '00501__B0_map_expiration_volume_2DMS_delTE2.3_e1_ph.nii.gz',
                      '00501__B0_map_expiration_volume_2DMS_delTE2.3_e2.nii.gz',
                      '00501__B0_map_expiration_volume_2DMS_delTE2.3_e2_ph.nii.gz',
                      '00501__B0_map_expiration_volume_2DMS_delTE2.3_e1_roi-R.nii.gz',
                      '00501__B0_map_expiration_volume_2DMS_delTE2.3_e1_roi-L.nii.gz',
                      ''])  # scanner-generated map not available for this dataset
FILENAMES_ALL.append(['00601__B0_map_expiration_volume_2DMS_delTE2.2_e1.nii.gz',  # philips_05
                      '00601__B0_map_expiration_volume_2DMS_delTE2.2_e1_ph.nii.gz',
                      '00601__B0_map_expiration_volume_2DMS_delTE2.2_e2.nii.gz',
                      '00601__B0_map_expiration_volume_2DMS_delTE2.2_e2_ph.nii.gz',
                      '00601__B0_map_expiration_volume_2DMS_delTE2.2_e1_roi-R.nii.gz',
                      '00601__B0_map_expiration_volume_2DMS_delTE2.2_e1_roi-L.nii.gz',
                      ''])  # scanner-generated map not available for this dataset
FILENAMES_ALL.append(['00701__B0_map_expiration_volume_2DMS_delTE2.46_e1.nii.gz',  # philips_06
                      '00701__B0_map_expiration_volume_2DMS_delTE2.46_e1_ph.nii.gz',
                      '00701__B0_map_expiration_volume_2DMS_delTE2.46_e2.nii.gz',
                      '00701__B0_map_expiration_volume_2DMS_delTE2.46_e2_ph.nii.gz',
                      '00701__B0_map_expiration_volume_2DMS_delTE2.46_e1_roi-R.nii.gz',
                      '00701__B0_map_expiration_volume_2DMS_delTE2.46_e1_roi-L.nii.gz',
                      ''])  # scanner-generated map not available for this dataset
FILENAMES_ALL.append(['00801__B0_map_expiration_volume_2DMS_delTE3_e1.nii.gz',  # philips_07
                      '00801__B0_map_expiration_volume_2DMS_delTE3_e1_ph.nii.gz',
                      '00801__B0_map_expiration_volume_2DMS_delTE3_e2.nii.gz',
                      '00801__B0_map_expiration_volume_2DMS_delTE3_e2_ph.nii.gz',
                      '00801__B0_map_expiration_volume_2DMS_delTE3_e1_roi-R.nii.gz',
                      '00801__B0_map_expiration_volume_2DMS_delTE3_e1_roi-L.nii.gz',
                      ''])  # scanner-generated map not available for this dataset
FILENAMES_ALL.append(['00901__B0_map_expiration_volume_2DMS_delTE3.45_e1.nii.gz',  # philips_08
                      '00901__B0_map_expiration_volume_2DMS_delTE3.45_e1_ph.nii.gz',
                      '00901__B0_map_expiration_volume_2DMS_delTE3.45_e2.nii.gz',
                      '00901__B0_map_expiration_volume_2DMS_delTE3.45_e2_ph.nii.gz',
                      '00901__B0_map_expiration_volume_2DMS_delTE3.45_e1_roi-R.nii.gz',
                      '00901__B0_map_expiration_volume_2DMS_delTE3.45_e1_roi-L.nii.gz',
                      ''])  # scanner-generated map not available for this dataset
FILENAMES_ALL.append(['01101__B0_map_expiration_volume_2DMS_delTEinphase_e1.nii.gz',  # philips_09
                      '01101__B0_map_expiration_volume_2DMS_delTEinphase_e1_ph.nii.gz',
                      '01101__B0_map_expiration_volume_2DMS_delTEinphase_e2.nii.gz',
                      '01101__B0_map_expiration_volume_2DMS_delTEinphase_e2_ph.nii.gz',
                      '01101__B0_map_expiration_volume_2DMS_delTEinphase_e1_roi-R.nii.gz',
                      '01101__B0_map_expiration_volume_2DMS_delTEinphase_e1_roi-L.nii.gz',
                      ''])  # scanner-generated map not available for this dataset
FILENAMES_ALL.append(['01201__B0_map_expiration_volume_2DMS_delTEoutphase_e1.nii.gz',  # philips_10
                      '01201__B0_map_expiration_volume_2DMS_delTEoutphase_e1_ph.nii.gz',
                      '01201__B0_map_expiration_volume_2DMS_delTEoutphase_e2.nii.gz',
                      '01201__B0_map_expiration_volume_2DMS_delTEoutphase_e2_ph.nii.gz',
                      '01201__B0_map_expiration_volume_2DMS_delTEoutphase_e1_roi-R.nii.gz',
                      '01201__B0_map_expiration_volume_2DMS_delTEoutphase_e1_roi-L.nii.gz',
                      ''])  # scanner-generated map not available for this dataset
FILENAMES_ALL.append(['02301__B0_map_expiration_volume_2DMS_delTE2.3_e1.nii.gz',  # philips_11
                      '02301__B0_map_expiration_volume_2DMS_delTE2.3_e1_ph.nii.gz',
                      '02301__B0_map_expiration_volume_2DMS_delTE2.3_e2.nii.gz',
                      '02301__B0_map_expiration_volume_2DMS_delTE2.3_e2_ph.nii.gz',
                      '02301__B0_map_expiration_volume_2DMS_delTE2.3_e1_roi-R.nii.gz',
                      '02301__B0_map_expiration_volume_2DMS_delTE2.3_e1_roi-L.nii.gz',
                      '02401__B0_map_expiration_volume_2DMS_product_e1a.nii.gz'])
FILENAMES_ALL.append(['02501__B0_map_expiration_volume_2DMS_delTE2.2_e1.nii.gz',  # philips_12
                      '02501__B0_map_expiration_volume_2DMS_delTE2.2_e1_ph.nii.gz',
                      '02501__B0_map_expiration_volume_2DMS_delTE2.2_e2.nii.gz',
                      '02501__B0_map_expiration_volume_2DMS_delTE2.2_e2_ph.nii.gz',
                      '02501__B0_map_expiration_volume_2DMS_delTE2.2_e1_roi-R.nii.gz',
                      '02501__B0_map_expiration_volume_2DMS_delTE2.2_e1_roi-L.nii.gz',
                      '02601__B0_map_expiration_volume_2DMS_product_2.2_e1a.nii.gz'])
FILENAMES_ALL.append(['02701__B0_map_expiration_volume_2DMS_delTE2.46_e1.nii.gz',  # philips_13
                      '02701__B0_map_expiration_volume_2DMS_delTE2.46_e1_ph.nii.gz',
                      '02701__B0_map_expiration_volume_2DMS_delTE2.46_e2.nii.gz',
                      '02701__B0_map_expiration_volume_2DMS_delTE2.46_e2_ph.nii.gz',
                      '02701__B0_map_expiration_volume_2DMS_delTE2.46_e1_roi-R.nii.gz',
                      '02701__B0_map_expiration_volume_2DMS_delTE2.46_e1_roi-L.nii.gz',
                      '02801__B0_map_expiration_volume_2DMS_product_2.46_e1a.nii.gz'])
FILENAMES_ALL.append(['02901__B0_map_expiration_volume_2DMS_delTE3_e1.nii.gz',  # philips_14
                      '02901__B0_map_expiration_volume_2DMS_delTE3_e1_ph.nii.gz',
                      '02901__B0_map_expiration_volume_2DMS_delTE3_e2.nii.gz',
                      '02901__B0_map_expiration_volume_2DMS_delTE3_e2_ph.nii.gz',
                      '02901__B0_map_expiration_volume_2DMS_delTE3_e1_roi-R.nii.gz',
                      '02901__B0_map_expiration_volume_2DMS_delTE3_e1_roi-L.nii.gz',
                      '03001__B0_map_expiration_volume_2DMS_product_3_e1a.nii.gz'])
FILENAMES_ALL.append(['03101__B0_map_expiration_volume_2DMS_delTE3.45_e1.nii.gz',  # philips_15
                      '03101__B0_map_expiration_volume_2DMS_delTE3.45_e1_ph.nii.gz',
                      '03101__B0_map_expiration_volume_2DMS_delTE3.45_e2.nii.gz',
                      '03101__B0_map_expiration_volume_2DMS_delTE3.45_e2_ph.nii.gz',
                      '03101__B0_map_expiration_volume_2DMS_delTE3.45_e1_roi-R.nii.gz',
                      '03101__B0_map_expiration_volume_2DMS_delTE3.45_e1_roi-L.nii.gz',
                      '03201__B0_map_expiration_volume_2DMS_product_3.45_e1a.nii.gz'])
FILENAMES_ALL.append(['03301__B0_map_expiration_volume_2DMS_delTEinphase_e1.nii.gz',  # philips_16
                      '03301__B0_map_expiration_volume_2DMS_delTEinphase_e1_ph.nii.gz',
                      '03301__B0_map_expiration_volume_2DMS_delTEinphase_e2.nii.gz',
                      '03301__B0_map_expiration_volume_2DMS_delTEinphase_e2_ph.nii.gz',
                      '03301__B0_map_expiration_volume_2DMS_delTEinphase_e1_roi-R.nii.gz',
                      '03301__B0_map_expiration_volume_2DMS_delTEinphase_e1_roi-L.nii.gz',
                      '03401__B0_map_expiration_volume_2DMS_product_3.45_e1a.nii.gz'])
FILENAMES_ALL.append(['03501__B0_map_expiration_volume_2DMS_delTEoutphase_e1.nii.gz',  # philips_17
                      '03501__B0_map_expiration_volume_2DMS_delTEoutphase_e1_ph.nii.gz',
                      '03501__B0_map_expiration_volume_2DMS_delTEoutphase_e2.nii.gz',
                      '03501__B0_map_expiration_volume_2DMS_delTEoutphase_e2_ph.nii.gz',
                      '03501__B0_map_expiration_volume_2DMS_delTEoutphase_e1_roi-R.nii.gz',
                      '03501__B0_map_expiration_volume_2DMS_delTEoutphase_e1_roi-L.nii.gz',
                      '03601__B0_map_expiration_volume_2DMS_product_1.76_e1a.nii.gz'])
FILENAMES_ALL.append(['0039_bh_b0map_default_bh_b0map_default_e1.nii.gz',  # siemens_02
                      '0040_bh_b0map_default_bh_b0map_default_e1_ph.nii.gz',
                      '0039_bh_b0map_default_bh_b0map_default_e2.nii.gz',
                      '0040_bh_b0map_default_bh_b0map_default_e2_ph.nii.gz',
                      '0039_bh_b0map_default_bh_b0map_default_e1_roi-R.nii.gz',
                      '0039_bh_b0map_default_bh_b0map_default_e1_roi-L.nii.gz',
                      ''])  # scanner-generated B0 map not available on Siemens
FILENAMES_ALL.append(['0041_bh_b0map_volume_bh_b0map_volume_e1.nii.gz',  # siemens_03
                      '0042_bh_b0map_volume_bh_b0map_volume_e1_ph.nii.gz',
                      '0041_bh_b0map_volume_bh_b0map_volume_e2.nii.gz',
                      '0042_bh_b0map_volume_bh_b0map_volume_e2_ph.nii.gz',
                      '0041_bh_b0map_volume_bh_b0map_volume_e1_roi-R.nii.gz',
                      '0041_bh_b0map_volume_bh_b0map_volume_e1_roi-L.nii.gz',
                      ''])  # scanner-generated B0 map not available on Siemens
FILENAMES_ALL.append(['00019__bh_b0map_default_e1.nii.gz',  # siemens_04
                      '00020__bh_b0map_default_e1_ph.nii.gz',
                      '00019__bh_b0map_default_e2.nii.gz',
                      '00020__bh_b0map_default_e2_ph.nii.gz',
                      '00019__bh_b0map_default_e1_roi-R.nii.gz',
                      '00019__bh_b0map_default_e1_roi-L.nii.gz',
                      ''])  # scanner-generated B0 map not available on Siemens
FILENAMES_ALL.append(['00029__bh_b0map_volume_e1.nii.gz',  # siemens_05
                      '00030__bh_b0map_volume_e1_ph.nii.gz',
                      '00029__bh_b0map_volume_e2.nii.gz',
                      '00030__bh_b0map_volume_e2_ph.nii.gz',
                      '00029__bh_b0map_volume_e1_roi-R.nii.gz',
                      '00029__bh_b0map_volume_e1_roi-L.nii.gz',
                      ''])  # scanner-generated B0 map not available on Siemens

# Define the main titles for the figure showing the unwrapping results for each dataset
SUPTITLES = []
SUPTITLES.append('ge_01 - 00720__Magnitude_ims__B0_map_dual_echo_e1: TEs (ms): [2.216, 5.136]')
SUPTITLES.append('philips_01 - 01001__B0_map_expiration_volume_2DMS_product_auto_e1 - default shim: TEs (ms): [4.001, 6.46]')
SUPTITLES.append('philips_02 - 01101__B0_map_expiration_volume_2DMS_product_auto_e1 - volume shim over kidneys: TEs (ms): [4.001, 6.46]')
SUPTITLES.append('philips_03 - 01301__B0_map_expiration_volume_2DMS_product_auto_e1 - volume shim over lungs: TEs (ms): [4.001, 6.46]')
SUPTITLES.append('siemens_01 - 0044_bh_b0map_fa3_default_bh_b0map_fa3_default_e1: TEs (ms): [4.000, 6.46]')
SUPTITLES.append('philips_04 - 00501__B0_map_expiration_volume_2DMS_delTE2.3: TEs (ms): [4.001, 6.3]')
SUPTITLES.append('philips_05 - 00601__B0_map_expiration_volume_2DMS_delTE2.2: TEs (ms): [4.001, 6.2]')
SUPTITLES.append('philips_06 - 00701__B0_map_expiration_volume_2DMS_delTE2.46: TEs (ms): [4.001, 6.46]')
SUPTITLES.append('philips_07 - 00801__B0_map_expiration_volume_2DMS_delTE3: TEs (ms): [4.001, 7.0]')
SUPTITLES.append('philips_08 - 00901__B0_map_expiration_volume_2DMS_delTE3.45: TEs (ms): [4.001, 7.45]')
SUPTITLES.append('philips_09 - 01101__B0_map_expiration_volume_2DMS_delTEinphase: TEs (ms): [4.001, 6.907]')
SUPTITLES.append('philips_10 - 01201__B0_map_expiration_volume_2DMS_delTEoutphase: TEs (ms): [4.001, 5.756]')
SUPTITLES.append('philips_11 - 02301__B0_map_expiration_volume_2DMS_delTE2.3: TEs (ms): [4.001, 6.3]')
SUPTITLES.append('philips_12 - 02501__B0_map_expiration_volume_2DMS_delTE2.2: TEs (ms): [4.001, 6.2]')
SUPTITLES.append('philips_13 - 02701__B0_map_expiration_volume_2DMS_delTE2.46: TEs (ms): [4.001, 6.46]')
SUPTITLES.append('philips_14 - 02901__B0_map_expiration_volume_2DMS_delTE3: TEs (ms): [4.001, 7.0]')
SUPTITLES.append('philips_15 - 03101__B0_map_expiration_volume_2DMS_delTE3.45: TEs (ms): [4.001, 7.45]')
SUPTITLES.append('philips_16 - 03301__B0_map_expiration_volume_2DMS_delTEinphase: TEs (ms): [4.001, 6.907]')
SUPTITLES.append('philips_17 - 03501__B0_map_expiration_volume_2DMS_delTEoutphase: TEs (ms): [4.001, 5.756]')
SUPTITLES.append('siemens_02 - 0039_bh_b0map_default_bh_b0map_default: TEs (ms): [4.001, 6.46]')
SUPTITLES.append('siemens_03 - 0041_bh_b0map_volume_bh_b0map_volume: TEs (ms): [4.001, 6.46]')
SUPTITLES.append('siemens_04 - 00019__bh_b0map_default: TEs (ms): [4.001, 6.46]')
SUPTITLES.append('siemens_05 - 00029__bh_b0map_volume: TEs (ms): [4.001, 6.46]')

TE_DIFFS = []
TE_DIFFS.append(5.136 - 2.216)  # ge_01
TE_DIFFS.append(6.46 - 4.001)   # philips_01
TE_DIFFS.append(6.46 - 4.001)   # philips_02
TE_DIFFS.append(6.46 - 4.001)   # philips_03
TE_DIFFS.append(6.46 - 4.000)   # siemens_01
TE_DIFFS.append(6.3 - 4.001)    # philips_04
TE_DIFFS.append(6.2 - 4.001)    # philips_05
TE_DIFFS.append(6.46 - 4.001)   # philips_06
TE_DIFFS.append(7.0 - 4.001)    # philips_07
TE_DIFFS.append(7.45 - 4.001)   # philips_08
TE_DIFFS.append(6.907 - 4.001)  # philips_09
TE_DIFFS.append(5.756 - 4.001)  # philips_10
TE_DIFFS.append(6.3 - 4.001)    # philips_11
TE_DIFFS.append(6.2 - 4.001)    # philips_12
TE_DIFFS.append(6.46 - 4.001)   # philips_13
TE_DIFFS.append(7.0 - 4.001)    # philips_14
TE_DIFFS.append(7.45 - 4.001)   # philips_15
TE_DIFFS.append(6.907 - 4.001)  # philips_16
TE_DIFFS.append(5.756 - 4.001)  # philips_17
TE_DIFFS.append(6.46 - 4.000)   # siemens_02
TE_DIFFS.append(6.46 - 4.000)   # siemens_03
TE_DIFFS.append(6.46 - 4.000)   # siemens_04
TE_DIFFS.append(6.46 - 4.000)   # siemens_05

# Normalise intensity range to the intensities of the kidney ROIs?
KIDNEY_INT_RANGES = False

for dataset, filenames, te_diff, slice_to_show, suptitle \
        in zip(DATASETS, FILENAMES_ALL, TE_DIFFS, SLICES_TO_SHOW, SUPTITLES):

    dir_dataset = os.path.join(DIR_DATA, dataset)
    filepaths = [os.path.join(dir_dataset, filename) for filename in filenames]

    unwrapping_results = CompareUnwrapping(filepaths, te_diff)

    # Plot maps
    unwrapping_results.plot_all(slice_to_show, suptitle, KIDNEY_INT_RANGES)

    # Print metrics within ROIs (used to generated unwrapping.xlsx)
    metrics = unwrapping_results.roi_metrics(slice_to_show)
    print(f"{dataset}: {metrics}")
