## [0.3.0-rc.2] - 2021-05-25

## Added 
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