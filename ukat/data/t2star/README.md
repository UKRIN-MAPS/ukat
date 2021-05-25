# Information about T2* (`t2star`) data

## Information common to all vendors

* All vendors can output (at least) magnitude and phase images. However, given that currently we are not using phase images in T2* processing, the only files that are uploaded to the subdirectories of `t2star/data` are magnitude images.
* 1 echo per NIfTI file

## GE-specific information

The scanner outputs magnitude, real and imaginary images, respectively.

## Philips-specific information

The scanner outputs magnitude and phase images, and a scanner calculated map.

## Siemens-specific information

The scanner outputs magnitude, magnitude (intensity normalised) and phase images, and a scanner calculated map.