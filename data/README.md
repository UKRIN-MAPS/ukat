# Information about test data

This document keeps track of the origin of the test data and other information relevant for analysis.

With the exception of datasets marked with :star:, all test data is from the travelling kidney pilot study 2019.

## BOLD R2*

* `r2star/ge`: subject 02, session 010, series 16. Contains magnitude, real and imaginary images. (**ANP checked**)
* `r2star/philips`: subject 02, session 002, series 14. Contains magnitude and phase images and scanner calculated map
* `r2star/siemens`: subject 02, session 008, series 24. Contains magnitude and phase images and scanner calculated map

## DWI

* `dwi/ge`: subject 04, session 005, series 14. (**ANP checked**)
* `dwi/philips`: subject 04, session 011, series 3901. (**Chosen to match subject from GE test data**)
* `dwi/siemens`<sup>[[1]](#siemens_bval_issue)</sup>: subject 04, session 009, series 42. (**Chosen to match subject from GE test data**)

## B0

* `b0/ge`: subject 004, session 005, series 9.
* `b0/philips_1`: subject 002, session 002, series 8. Contains scanner calculated B0 map, 1 phase and 1 magnitude image (from a single echo)
* :star: `b0/philips_2`: Data acquired with phase and magnitude data saved for both echoes

## Notes

<a name="siemens_bval_issue"><sup>[1]</sup></a> There is a limitation in Siemens `*.bval` and `*.bvec` files: b-values are rounded to multiples of 50. This is an issue for us because we use several low b-values which are not multiples of 50. Furthermore, if the result of the rounding is a b-value of 0, the corresponding b-vector becomes [0, 0, 0] (see table below). Inspecting the data does suggest that the prescribed b-values are indeed used for the measurements. Therefore, it seems we can use the low b-value measurements in the analysis provided that the `.bval` file is corrected to account for this issue.

| Measurement # | Prescribed b-value | b-value on `*.bval` file | Corresponding b-vector is affected |
|:---:|:---:|:---:|:---:|
| 1 | 0 | 0 | No |
| 2 | 0 | 0 | No |
| 3 | 5 | **0** | **Yes** |
| 4 | 10 | **0** | **Yes** |
| 5 | 20 | **0** | **Yes** |
| 6 | 30 | **50** | **Unclear** |
| 7 | 40 | **50** | **Unclear** |
| 8 | 50 | 50 | No |
| 9 | 70 | **50** | **Unclear** |
| 10 | 100 | 100 | No |

As a result of this, the Siemens dataset contains two b-value files:

1. `00042__trig_dwi_13b_06dir.bval_before_manual_correction`: original `*.bval` file obtained directly from DICOM to NIfTI conversion with [`d2n`](https://github.com/UKRIN-MAPS/d2n).
2. `00042__trig_dwi_13b_06dir.bval`: corrected `*.bval` file to use for analysis.

An approach for correcting the `*.bvec` file is currently not implemented, but given that we do not rely on gradient direction information for the analyses we have currently planned (monoexponential fit or IVIM), this is not an issue. Furthermore, it is likely that any future protocols for DTI will not include such low b-values.
