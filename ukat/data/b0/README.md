# Information about B<sub>0</sub> mapping (`b0`) data

## Information common to all vendors

## GE-specific information
 * magnitude, real and imaginary data, respectively, saved for both echoes
 * 1 echo per NIfTI file

## Philips-specific information

* `b0/philips_1`: contains scanner calculated B0 map, 1 phase and 1 magnitude image (from a single echo)
* `b0/philips_2`: phase and magnitude data saved for both echoes

## Siemens-specific information

* Two series numbers corresponding to magnitude and phase data, respectively.
* 1 echo per NIfTI file