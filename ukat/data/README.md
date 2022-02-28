# General information about test data

This document contains two types of information:

1. Origin of test data
2. Information which is general enough to apply to all types of MR test data

Specific information for each type of MR data lives on the README.md files in each data subdirectory: [`b0`](b0/README.md), [`dwi`](dwi/README.md), [`t1`](t1/README.md), [`t2`](t2/README.md) and [`t2star`](t2star/README.md). The data itself is in online storage, we recommend the use of Zenodo and have a [UKRIN community](https://zenodo.org/communities/ukrin/) to help keep data curated. If you want to add data to this community please use [this link](https://zenodo.org/deposit/new?c=ukrin).

## Origin of test data

With the exception of datasets marked with :star:, all test data is from the travelling kidney pilot study 2019.

### BOLD R2*

* `t2star/ge`: subject 02, session 010, series 16
* `t2star/philips`: subject 02, session 002, series 14
* `t2star/siemens`: subject 02, session 007, series 24/25/26/27
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
* `t1/philips_1`: NIST Phantom
* `t1/philips_2`: not part of travelling kidney pilot study 2019, data acquired on 20 Oct 2020 in Nottingham
* `t1_molli_philips`: dataset collected as part of the second travelling kidney study on 27/10/2021 in Nottingham 

### T2
* `t2/philips_1`: not part of travelling kidney pilot study 2019, data acquired on 06 Aug 2020 in Nottingham

### MT
* `mt/philips`: Part of travelling kidney study 2021, data acquired on 05 July 2021 in Nottingham

### Phase Contrast
* `phase_contrast/philips_left`: Part of travelling kidney study 2021, data acquired on 05th July 2021 in Nottingham. Subject 004, series 1901.
* `phase_contrast/philips_right`: Part of travelling kidney study 2021, data acquired on 05th July 2021 in Nottingham. Subject 004, series 2001.