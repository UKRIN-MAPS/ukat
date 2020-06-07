# Information about diffusion-weighted imaging (`dwi`) data

## Info about "box" masks

:warning:
We all need to discuss and agree on the preprocessing approach for `dwi` data. The current preprocessing approach (_work-in-progress_) uses "box" masks but this may change in the future.
:warning:

"Box" masks are used to specify the regions-of-interest to undergo intrasubject registrations, separate for right and left kidneys, for motion correction of time series data. These masks are stored in two files (one for each kidney) called `box_01.nii.gz` and `box_02.nii.gz`. This document provides some info about how @fnery generates these masks. A `box_check.png` image is also available for each `dwi` test dataset to quickly visualize these masks.

### Drawing box masks

For reference, @fnery uses [`mrview`](https://mrtrix.readthedocs.io/en/latest/reference/commands/mrview.html) to draw these masks. This is just a matter of personal preference and other programs can also be used for this purpose (e.g. [`fsleyes`](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes), [`ITK-SNAP`](http://www.itksnap.org/pmwiki/pmwiki.php)).

#### Displaying the images for drawing masks

```bash
$ mrview siemens/00042__trig_dwi_13b_06dir.nii.gz --interpolation 0 -mode 2 -size 1920,1000 -position 0,27 &
$ mrview ge/00014__Cor_DWI_RT.nii.gz --interpolation 0 -mode 2 -size 1920,1000 -position 0,27
$ mrview philips/03901__DWI_5slices.nii.gz --interpolation 0 -mode 2 -size 1920,1000 -position 0,27 &
```

#### Generating `box_check.png` files

```bash
$ mrview siemens/00042__trig_dwi_13b_06dir.nii.gz -roi.load siemens/box_01.nii.gz -roi.opacity 0.07 -roi.colour 1,0,0 -roi.load siemens/box_02.nii.gz -roi.opacity 0.07 -roi.colour 0,1,0 &
$ mrview ge/00014__Cor_DWI_RT.nii.gz -roi.load ge/box_01.nii.gz -roi.opacity 0.07 -roi.colour 1,0,0 -roi.load ge/box_02.nii.gz -roi.opacity 0.07 -roi.colour 0,1,0 &
$ mrview philips/03901__DWI_5slices.nii.gz -roi.load philips/box_01.nii.gz -roi.opacity 0.07 -roi.colour 1,0,0 -roi.load philips/box_02.nii.gz -roi.opacity 0.07 -roi.colour 0,1,0 &
```

And then:

* Select lightbox view
* Make sure the slice increment (mm) is set to the slice thickness (+ gap if applicable)
* Zoom to kidneys (mousewheel)

## Information common to all vendors

All data lives on a single 4D NIfTI file. Diffusion gradients information is stored in `*.bval` and `*.bvec` files generated automatically when converting to NIfTI (with [`d2n`](https://github.com/UKRIN-MAPS/d2n)).

## GE-specific information

## Philips-specific information

## Siemens-specific information

### Limitation in `*.bval` and `*.bvec` files

b-values are rounded to multiples of 50. This affects us because we use several low b-values which are not multiples of 50. Furthermore, if the result of the rounding is a b-value of 0, the corresponding b-vector becomes [0, 0, 0] (see table below). Inspecting the data does suggest that the prescribed b-values are indeed used for the measurements. Therefore, it seems we can use the low b-value measurements in the analysis provided that the `.bval` file is corrected to account for this issue.

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
