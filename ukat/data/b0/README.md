# Information about B<sub>0</sub> mapping (`b0`) data

## GE-specific information
 * magnitude, real and imaginary data, respectively, saved for both echoes
 * 1 echo per NIfTI file

## Philips-specific information

* `b0/philips_1`: contains scanner calculated B0 map, 1 phase and 1 magnitude image (from a single echo)
* `b0/philips_2`: phase and magnitude data saved for both echoes
* `b0/philips_phantom`: magnitude, phase, real and imaginary data of a NIST phantom saved for both echoes. The purpose is to calculate the phase images from the real and imaginary data and compare with the phase that comes from the scanner.
* `b0/philips_phantom_productb0map`: product B0 Map magnitude and product B0 Map phase images of a NIST phantom

## Siemens-specific information

* Two series numbers corresponding to magnitude and phase data, respectively.
* 1 echo per NIfTI file