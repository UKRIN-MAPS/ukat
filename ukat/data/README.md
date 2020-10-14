# General information about test data

This document contains two types of information:

1. Origin of test data
2. Information which is general enough to apply to all types of MR test data

Specific information for each type of MR data lives on the README.md files in each data subdirectory: [`b0`](b0/README.md), [`dwi`](dwi/README.md), [`r2star`](r2star/README.md).

## Origin of test data

With the exception of datasets marked with :star:, all test data is from the travelling kidney pilot study 2019.

### BOLD R2*

* `r2star/ge`: subject 02, session 010, series 16
* `r2star/philips`: subject 02, session 002, series 14
* `r2star/siemens`: subject 02, session 007, series 24/25/26/27
### DWI

* `dwi/ge`: subject 04, session 005, series 14. (**ANP checked**)
* `dwi/philips`: subject 04, session 011, series 3901. (**Chosen to match subject from GE test data**)
* `dwi/siemens`: subject 04, session 009, series 42. (**Chosen to match subject from GE test data**)

### B0

* `b0/ge`: subject 004, session 005, series 9
* `b0/philips_1`: subject 002, session 002, series 8
* :star: `b0/philips_2`: not part of travelling kidney pilot study 2019
* `b0/siemens_1`: subject 004, session 009, series 010/011
* :star: `b0/siemens_2`: not part of travelling kidney pilot study 2019, data acquired after GOSH Prisma upgrade to VE11E (20200305_0281)

### T1
* `t1/philips`: NIST Phantom