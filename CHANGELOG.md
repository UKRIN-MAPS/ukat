## [0.6.2] - 2022-10-18

### Added
* Rescaling of GE B1 maps. #190

### Changed

### Fixed
* Hopefully reference manual now works, built using Python 3.9. #189

## [0.6.1] - 2022-08-03

### Added
* Whole kidney segmentation can now be performed using custom models. #187

### Changed
* Python 3.7 is no longer supported.

### Fixed

## [0.6.0] - 2022-02-28

### Added
* Phase contrast analysis #169 #171
* MOLLI T1 mapping #179

### Changed

### Fixed
* Bugs in MTR #182
* Segmentation import #184

## [0.5.0] - 2022-01-11

### Added
* Automated kidney segmentation #163 #175
* iSNR calculations #162
* tSNR calculations #160 #161
* Reference manual is now available [online](https://ukrin-maps.github.io/ukat/) #173 #174

### Changed
* B0 offset minimisation #42 #172
* Single slice T1 mapping is now possible #177
* Removed redundant directories #176
* Codecov updated #167 #168
* Bump Python requirement to 3.7

### Fixed
* Minor formatting/function order fixes


## [0.4.0] - 2021-08-05

### Added 
* Magnetisation Transfer Ratio (MTR) mapping #156 #153

### Changed
* ADC fitting now uses the full complement of supplied b-values to fit each voxel (or ignores a voxel if any signal is not positive) #157 #158
* Removed infinite values from R2* maps #154 #155

### Fixed
* Bump dipy requirement to a minimum of v1.2.0 #152 #155
* Minor documentation typos


## [0.3.0] - 2021-05-25

### Added 
* PyPI release badge to readme

### Changed
* Data is now stored externally and downloaded at runtime #147 #148 #62
* Manifest added to enable more specific packaging for pypi
* New release action generates pre-release for tags on branches other than master

### Fixed
* Link to dipy contribting guidelines

## [0.2.2] - 2021-05-11

### Added
* `ukat` is now available on PyPI. Simply run `pip install ukat`


## [0.2.0] - 2021-05-10

### Added
* `ukat` is now public under GPL3 #120
* ADC mapping (monoexponential and MD models) #136 #137 #23
* T1 mapping #80 #82
* Additional T2 fitting models (`3p_exp` three parameter exponential fit and excluding signal below a supplied threshold) #134
* `to_nifit` method added to all mapping classes for quick export of calculated maps #119

### Changed
* T2* confidence Intervals are now returned for `2p_exp` fitting model #112 #141
* T2 fitting upper bound from 700 ms to 1000 ms #116
* Tests now include an end to end test with "real" data acquired on a scanner rather than just simulated data #128
* Data fetched refactored #143
* Removed support for Python 3.5

### Fixed
* Consistent use of `npt` rather than `npt` and `np.testing` #96


## [0.1.0] - 2020-10-28

### Added
* B0 mapping #58 #50 #51 #33 #30 #34 #32 #29 #15
* T2* mapping #17 #55 #40 #16
* General project structure

### Changed
* First release so no previous release to change from

### Fixed
* First release so no previous release to fix from