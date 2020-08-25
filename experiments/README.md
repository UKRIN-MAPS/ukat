# Experiments

:warning: This directory and its contents are experimental and may be removed from `ukat` at any point.

## Difference between experiments and tutorials

- `tutorials` directory: contains code strictly for demonstration main features of `ukat`.
- `experiments` directory: contains code (and associated data if relevant) that supports decisions taken in the implementation of `ukat` tools, that doesn't need to be part of the `ukat` package but which we want to make reproducible.

## List of experiments

### `philips_b0_validate_phase_calculation`

- Purpose: Demonstrate calculation of magnitude and phase from real and imaginary data that matches magnitude and phase data calculated at the scanner.
- Data: 3 B0 mapping datasets with different shims. Each dataset contains real, imaginary, magnitude and phase data.
- _Note for @fnery: The files in the data directory are a subset of dataset 20200820 (see log:20200823)_.

### `b0_unwrap_and_compare_with_scanner_calculated_map`
- Purpose: Demonstrate calculation B0 map (including different unwrapping methods) and comparison with scanner-calculated B0 map.
- Data:
    - 1 GE dataset containing magnitude and phase data, as well as scanner-calculated B0 map (all from the same breath-hold). Subset of data from SUBJECT:cam004>>MRSESSION:cam004_MR_1 in [UKRIN test XNAT instance](https://test-ukrin.dpuk.org).
    - 3 Philips datasets:
        - `philips_1`: "default shim"
        - `philips_2`: "volume shim over kidneys"
        - `philips_3`: "volume shim over lungs"
        - Note that each dataset contains two scans (i.e. two breath-holds): one to generate the magnitude & phase images and another to generate the scanner-calculated map.
        - _Note for @fnery: These files are a subset of dataset 20200820 (see log:20200825)_.
    - Details on the full file tree below (excluding .json files):
    ```bash
        ├── ge_1
        │   ├── 00720__Magnitude_ims__B0_map_dual_echo_e1.nii.gz  # echo 1 magnitude
        │   ├── 00720__Magnitude_ims__B0_map_dual_echo_e2.nii.gz  # echo 2 magnitude
        │   ├── 00721__Phase_ims__B0_map_dual_echo_e1.nii.gz      # echo 1 phase
        │   ├── 00721__Phase_ims__B0_map_dual_echo_e2.nii.gz      # echo 2 phase
        │   └── 00730__B0_off-resonance_maps_(Hz_x_10).nii.gz     # scanner-calculated B0 map
        ├── philips_1
        │   ├── 00501__B0_map_expiration_volume_2DMS_product_auto_e1.nii.gz     # (won't be used) magnitude corresponding to series with scanner-calculated b0 map
        │   ├── 00501__B0_map_expiration_volume_2DMS_product_auto_e1a.nii.gz    # scanner-calculated B0 map
        │   ├── 01001__B0_map_expiration_volume_2DMS_product_auto_e1.nii.gz     # echo 1 magnitude
        │   ├── 01001__B0_map_expiration_volume_2DMS_product_auto_e1_ph.nii.gz  # echo 1 phase
        │   ├── 01001__B0_map_expiration_volume_2DMS_product_auto_e2.nii.gz     # echo 2 magnitude
        │   └── 01001__B0_map_expiration_volume_2DMS_product_auto_e2_ph.nii.gz  # echo 2 phase
        ├── philips_2
        │   ├── 00701__B0_map_expiration_volume_2DMS_product_auto_e1.nii.gz     # (won't be used) magnitude corresponding to series with scanner-calculated b0 map
        │   ├── 00701__B0_map_expiration_volume_2DMS_product_auto_e1a.nii.gz    # scanner-calculated B0 map
        │   ├── 01101__B0_map_expiration_volume_2DMS_product_auto_e1.nii.gz     # echo 1 magnitude
        │   ├── 01101__B0_map_expiration_volume_2DMS_product_auto_e1_ph.nii.gz  # echo 1 phase
        │   ├── 01101__B0_map_expiration_volume_2DMS_product_auto_e2.nii.gz     # echo 2 magnitude
        │   └── 01101__B0_map_expiration_volume_2DMS_product_auto_e2_ph.nii.gz  # echo 2 phase
        └── philips_3

            ├── 01201__B0_map_expiration_volume_2DMS_product_auto_e1.nii.gz     # (won't be used) magnitude corresponding to series with scanner-calculated b0 map

            ├── 01201__B0_map_expiration_volume_2DMS_product_auto_e1a.nii.gz    # scanner-calculated B0 map

            ├── 01301__B0_map_expiration_volume_2DMS_product_auto_e1.nii.gz     # echo 1 magnitude

            ├── 01301__B0_map_expiration_volume_2DMS_product_auto_e1_ph.nii.gz  # echo 1 phase

            ├── 01301__B0_map_expiration_volume_2DMS_product_auto_e2.nii.gz     # echo 2 magnitude

            └── 01301__B0_map_expiration_volume_2DMS_product_auto_e2_ph.nii.gz  # echo 2 phase
        ```

