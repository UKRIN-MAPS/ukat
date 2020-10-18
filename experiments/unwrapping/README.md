# Unwrapping experiment

## Purpose
Purpose: compare two unwrapping methods:
- `unwrap_phase` from the `restoration` module from `scikit-image`
- `PRELUDE` from FSL, invoked here via `fslpy`

There are 23 test datasets, each stored in subdirectory of `data` (1 GE, 17 Philips, 5 Siemens). Data is in NIfTI format. The `.json` files aren't used directly but are kept in the `data` directory for reference.

Data sources:
- `ge_01`: Subset of data from SUBJECT:cam004>>MRSESSION:cam004_MR_1 in [UKRIN test XNAT instance](https://test-ukrin.dpuk.org).
- `philips_01`: Acquired by @charlotteebuchanan, sent to @fnery for processing (subset of 20200820 (see log:20200825))
- `philips_02`: Acquired by @charlotteebuchanan, sent to @fnery for processing (subset of 20200820 (see log:20200825))
- `philips_03`: Acquired by @charlotteebuchanan, sent to @fnery for processing (subset of 20200820 (see log:20200825))
- `siemens_01`: Acquired by @fnery (subset of 20200305_0281)
- `philips_04`: Acquired by @charlotteebuchanan, sent to @fnery for processing (subset of 20200904_b0_nottingham)
- `philips_05`: Acquired by @charlotteebuchanan, sent to @fnery for processing (subset of 20200904_b0_nottingham)
- `philips_06`: Acquired by @charlotteebuchanan, sent to @fnery for processing (subset of 20200904_b0_nottingham)
- `philips_07`: Acquired by @charlotteebuchanan, sent to @fnery for processing (subset of 20200904_b0_nottingham)
- `philips_08`: Acquired by @charlotteebuchanan, sent to @fnery for processing (subset of 20200904_b0_nottingham)
- `philips_09`: Acquired by @charlotteebuchanan, sent to @fnery for processing (subset of 20200904_b0_nottingham)
- `philips_10`: Acquired by @charlotteebuchanan, sent to @fnery for processing (subset of 20200904_b0_nottingham)
- `philips_11`: Acquired by @charlotteebuchanan, sent to @fnery for processing (subset of 20201006_b0_nottingham)
- `philips_12`: Acquired by @charlotteebuchanan, sent to @fnery for processing (subset of 20201006_b0_nottingham)
- `philips_13`: Acquired by @charlotteebuchanan, sent to @fnery for processing (subset of 20201006_b0_nottingham)
- `philips_14`: Acquired by @charlotteebuchanan, sent to @fnery for processing (subset of 20201006_b0_nottingham)
- `philips_15`: Acquired by @charlotteebuchanan, sent to @fnery for processing (subset of 20201006_b0_nottingham)
- `philips_16`: Acquired by @charlotteebuchanan, sent to @fnery for processing (subset of 20201006_b0_nottingham)
- `philips_17`: Acquired by @charlotteebuchanan, sent to @fnery for processing (subset of 20201006_b0_nottingham)
- `siemens_02`: Acquired by @fnery (subset of 20200310_0284)
- `siemens_03`: Acquired by @fnery (subset of 20200310_0284)
- `siemens_04`: Acquired by @fnery (subset of 20200927_0288)
- `siemens_05`: Acquired by @fnery (subset of 20200927_0288)

Note that `philips_11`...`philips_17` include the same acquisitions in `philips_04`...`philips_10` (performed on a different day) with the exception that `philips_11`...`philips_17` also include scanner calculated B0 maps (acquired with the same acquisition parameters but from a separate breath-hold).

## Running the code
The main file to run is `unwrapping_main.py`. The other `.py` files contain general tools used by `unwrapping_main.py`.

Dependencies:
- `ukat`
- FSL and `fslpy`
- `arraystats.py` and `roi.py`

If any files are moved or renamed, `unwrapping_main.py` will have to be updated accordingly (file paths are relative).

## Output
Running `unwrapping_main.py` will generate the figure montages show in `unwrapping.pptx`. In addition, metrics for each dataset will be printed in your terminal which were pasted into `unwrapping.xlsx` for further analysis. This table is also in `unwrapping.pptx`.