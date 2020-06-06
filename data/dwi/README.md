# Info about "box" masks



:warning:
This document contains some information about "box" masks. This is the sort of thing that (when we all agree on the particular approach) should go to a SOP document/repository so this might not be the ideal place to store this information but for now it should work.
:warning:

"Box" masks are used to specify the regions-of-interest to undergo intrasubject registrations, separate for right and left kidneys, for motion correction of time series data. These masks are stored in two files (one for each kidney) called `box_01.nii.gz` and `box_02.nii.gz`. This document provides some info about how @fnery generates these masks. A `box_check.png` image is also available for each `dwi` test dataset to quickly visualize these masks.

## Drawing box masks

For reference, @fnery uses [`mrview`](https://mrtrix.readthedocs.io/en/latest/reference/commands/mrview.html) to draw these masks. This is just a matter of personal preference and other programs can also be used for this purpose (e.g. [`fsleyes`](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes), [`ITK-SNAP`](http://www.itksnap.org/pmwiki/pmwiki.php)).

### Displaying the images for drawing masks

```bash
$ mrview siemens/00042__trig_dwi_13b_06dir.nii.gz --interpolation 0 -mode 2 -size 1920,1000 -position 0,27 &
$ mrview ge/00014__Cor_DWI_RT.nii.gz --interpolation 0 -mode 2 -size 1920,1000 -position 0,27
$ mrview philips/03901__DWI_5slices.nii.gz --interpolation 0 -mode 2 -size 1920,1000 -position 0,27 &
```

### Generating `box_check.png` files:

```bash
$ mrview siemens/00042__trig_dwi_13b_06dir.nii.gz -roi.load siemens/box_01.nii.gz -roi.opacity 0.07 -roi.colour 1,0,0 -roi.load siemens/box_02.nii.gz -roi.opacity 0.07 -roi.colour 0,1,0 &
$ mrview ge/00014__Cor_DWI_RT.nii.gz -roi.load ge/box_01.nii.gz -roi.opacity 0.07 -roi.colour 1,0,0 -roi.load ge/box_02.nii.gz -roi.opacity 0.07 -roi.colour 0,1,0 &
$ mrview philips/03901__DWI_5slices.nii.gz -roi.load philips/box_01.nii.gz -roi.opacity 0.07 -roi.colour 1,0,0 -roi.load philips/box_02.nii.gz -roi.opacity 0.07 -roi.colour 0,1,0 &
```

And then:

* Select lightbox view
* Make sure the slice increment (mm) is set to the slice thickness (+ gap if applicable)
* Zoom to kidneys (mousewheel)