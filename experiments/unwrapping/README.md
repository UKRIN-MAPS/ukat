# Unwrapping experiment

## Purpose
Purpose: directly compare two unwrapping methods:
- `unwrap_phase` from the `restoration` module from `scikit-image`
- `PRELUDE` from FSL, invoked here via `fslpy`

There are 12 test datasets, each stored in subdirectory of `data`:
```
Dataset    | Figure title                                                                    | TEs (ms)
-----------------------------------------------------------------------------------------------------------
ge_01      | 00720__Magnitude_ims__B0_map_dual_echo_e1                                       | 2.216, 5.136
philips_01 | 01001__B0_map_expiration_volume_2DMS_product_auto_e1 (default shim)             | 4.001, 6.46
philips_02 | 01101__B0_map_expiration_volume_2DMS_product_auto_e1 (volume shim over kidneys) | 4.001, 6.46
philips_03 | 01301__B0_map_expiration_volume_2DMS_product_auto_e1 (volume shim over lungs)   | 4.001, 6.46
siemens_01 | 0044_bh_b0map_fa3_default_bh_b0map_fa3_default_e1                               | 4.000, 6.46
philips_04 | 00501__B0_map_expiration_volume_2DMS_delTE2.3                                   | 4.001, 6.3
philips_05 | 00601__B0_map_expiration_volume_2DMS_delTE2.2                                   | 4.001, 6.2
philips_06 | 00701__B0_map_expiration_volume_2DMS_delTE2.46                                  | 4.001, 6.46
philips_07 | 00801__B0_map_expiration_volume_2DMS_delTE3                                     | 4.001, 7.0
philips_08 | 00901__B0_map_expiration_volume_2DMS_delTE3.45                                  | 4.001, 7.45
philips_09 | 01101__B0_map_expiration_volume_2DMS_delTEinphase                               | 4.001, 6.907
philips_10 | 01201__B0_map_expiration_volume_2DMS_delTEoutphase                              | 4.001, 5.756
-----------------------------------------------------------------------------------------------------------
```

Each "dataset" in this experiment is composed of 10 files:
- magnitude TE#1 (`.nii.gz` and `.json`)
- magnitude TE#2 (`.nii.gz` and `.json`)
- phase TE#1 (`.nii.gz` and `.json`)
- phase TE#2 (`.nii.gz` and `.json`)
- ROI kidney 1 (`.nii.gz`)
- ROI kidney 2 (`.nii.gz`)

The `.json` files aren't used directly but are kept in the `data` directory for reference.

Data sources:
- `ge_01`: Subset of data from SUBJECT:cam004>>MRSESSION:cam004_MR_1 in [UKRIN test XNAT instance](https://test-ukrin.dpuk.org).
- `philips_01`: Acquired by @charlotteebuchanan, sent to @fnery for processing:arrow_right:
- `philips_02`: Acquired by @charlotteebuchanan, sent to @fnery for processing:arrow_right:
- `philips_03`: Acquired by @charlotteebuchanan, sent to @fnery for processing:arrow_right:
- `siemens_01`: Acquired by @fnery:arrow_right::arrow_right:
- `philips_04`: Acquired by @charlotteebuchanan, sent to @fnery for processing:arrow_right:
- `philips_05`: Acquired by @charlotteebuchanan, sent to @fnery for processing:arrow_right:
- `philips_06`: Acquired by @charlotteebuchanan, sent to @fnery for processing:arrow_right:
- `philips_07`: Acquired by @charlotteebuchanan, sent to @fnery for processing:arrow_right:
- `philips_08`: Acquired by @charlotteebuchanan, sent to @fnery for processing:arrow_right:
- `philips_09`: Acquired by @charlotteebuchanan, sent to @fnery for processing:arrow_right:
- `philips_10`: Acquired by @charlotteebuchanan, sent to @fnery for processing:arrow_right:

:arrow_right: _Note for @fnery:_ These files are A subset of dataset 20200820 (see log:20200825). \
:arrow_right::arrow_right: _Note for @fnery_: These files are a subset of dataset 20200305_0281.

## Running the code
The main file to run is `unwrapping.py`. The other `.py` files contain general tools used by `unwrapping.py`.

Dependencies:
- `ukat`
- FSL and `fslpy`
- `arraystats.py` and `roi.py`

If any files are moved or renamed, `unwrapping.py` will have to be updated accordingly.

## Output
Running `unwrapping.py` will generate 12 figure montages (one per dataset). These are also saved in `unwrapping.pptx`.