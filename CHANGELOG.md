## [0.7.3] - 2024-10-30

### Changed
* Signals to input to mapping classes are now normalised before fitting, then M0 rescaled back to the original scale. This 
  means M0 bounds and initialisation are more consistent/appropriate for data from different vendors #226
* The limits of efficiency have been widened for MOLLI T1 mapping #229
* Upgrade codecov action to v4

## [0.7.2] - 2024-07-05

### Added
* Export the fit signals from mapping functions e.g. the expected T1 recovery at times TI given the fit values of T1 and M0. #221
* Warnings if there aren't many negative values in T1 mapping data. #222 #223
* `get_fit_signal` methods have been added to most mapping classes to allow the user to get the fit signal at a given time point.

### Fixed
* Issue where resources sub-module was not included on PyPI

### Changed
* T1 mapping data is now assumed to have been magnitude corrected if the first percentile of the first inversion time is 
  negative rather than the minimum value. This should make it more robust to noise/preprocessing artefacts. #222 #223

## [0.7.1] - 2024-02-23

### Added
* B1 rescaling as function and default output of StimFit #218

### Changed
* `ukat` is now tested on MacOS #219

### Fixed
* DWI mask export bug #217

## [0.7.0] - 2023-09-07

### Added
* T2StimFit - A method of accounting for stimulated echoes when performing T2 mapping #207, #209
* R-Squared values for curve fitting maps #198 #205
* New PR template for releases #205 #210

### Changed
* `ukat` is now tested against Python 3.11
* Dependencies are now a little less strict for some packages

### Fixed
* Mapping should now scale better over large images/multiple cores #165 #205
* Quite a lot of PEP8 formatting issues

## [0.6.5] - 2023-02-17

### Added
* Number of downloads badge to readme

### Changed
* Lots of dependencies have been updated to their latest versions #203
* Dependencies are now set to be compatible versions (~=) rather than greater than a specific version (>=) #203

### Fixed
* MTR outputs are now squeezed to remove bonus dimensions #202

## [0.6.4] - 2022-11-21

### Added
* Python 3.10 is now supported.

### Changed
* Increased upper bounds of M0 in T1, T2 and T2* exponential fit. #196 #199
* B0 mask now keeps True voxels rather than False voxels, making it align with the rest of UKAT. #194 #195

### Fixed
* Fixed bug in B0 offset calculation. #200

## [0.6.3] - 2022-10-26

### Added
* iSNR maps #192

### Changed

### Fixed

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