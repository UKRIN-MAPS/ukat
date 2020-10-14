"""This module implements the ArrayStats class which calculates several
descriptive statistics of 2, 3 or 4D input arrays. Statistics are calculated
for all possible dimensions of input array except the first dimension (medical
images were the use case in mind when implementing this).

That is, if the input array is a 4D image (rows, columns, slices, "time"), it
calculates statistics for each 2D slice, each 3D volume and each 4D volume.
See docstring of the `calculate` method for the list of the calculated
statistical measures.

"""
import numpy as np
from scipy import stats

NOT_CALCULATED_MSG = 'Not calculated. See ArrayStats.calculate().'

class ArrayStats():
    """Class to calculate array statistics (optionally within a mask)

    Parameters
    ----------
    image : np.ndarray
        array with 2, 3 or 4 dimensions
    roi : np.ndarray (dtype: bool), optional
        region of interest, same dimensions as `image`

    Attributes
    ----------
    image : see above (parameters)
    roi : see above (parameters)
    image_shape : tuple
        shape (length of each dimension of image)
    image_ndims: int
        number of dimensions of image

    """
    def __init__(self, image, roi=None):
        """ Init method: see class docstring for parameters/attributes

        """
        image_shape = image.shape
        image_ndims = len(image_shape)

        # Error checks
        if image_ndims < 2 or image_ndims > 4:
            raise ValueError("`image` must be [2, 3, 4]D")

        # if image_ndims == 4:
        #     raise RuntimeError("`ArrayStats` hasn't been validated for 4D arrays, so do that before trying to use it")

        # Initialise unspecified input arguments
        if roi is None:
            roi = np.ones(image_shape, dtype=bool)
        elif isinstance(roi, np.ndarray) and roi.dtype == bool:
            if roi.shape != image_shape:
                raise ValueError("`roi.shape` must match `image.shape`")
        else:
            raise TypeError("`roi` must None or a numpy array with dtype=bool")

        self.image = image
        self.roi = roi
        self.image_shape = image_shape
        self.image_ndims = image_ndims

    def calculate(self):
        """Calculate array statistics

        Returns
        -------
        dict
            dictionary where the keys are the calculated statistical measures:
            - 'n'        : number of array elements
            - 'mean'
            - 'median'
            - 'min'
            - 'max'
            - 'std'      : standard deviation
            - 'cv'       : coefficient of variation (std/mean)
            - 'skewness'
            - 'kurtosis'
            Each of the statistical measures is a dictionary in itself, where
            keys are the dimensions over which the calculation was performed.
            For example:
            - statistics['mean']['2D']: mean of each 2D slice
            - statistics['mean']['3D']: mean of each 3D volume
            - statistics['mean']['4D']: mean of each 4D volume

        """
        # Init attribute that may need to be modified (i.e. add new axis)
        image = self.image
        roi = self.roi

        # Add new axis to `image` and `roi` if needed
        if self.image_ndims == 2:
            image = image[:, :, np.newaxis, np.newaxis]
            roi = roi[:, :, np.newaxis, np.newaxis]
        elif self.image_ndims == 3:
            image = image[:, :, :, np.newaxis]
            roi = roi[:, :, :, np.newaxis]

        # Get number of slices and time points of expanded `image` (i.e. after
        # np.newaxis). Note the `image_shape` attribute initialised in __init__
        # refers to the shape before adding new axis
        (_, _, nz, nt) = image.shape

        # Pre-allocate 2D
        tmp2 = np.full((nz, nt), np.nan)
        n2 = np.copy(tmp2)
        mean2 = np.copy(tmp2)
        median2 = np.copy(tmp2)
        min2 = np.copy(tmp2)
        max2 = np.copy(tmp2)
        std2 = np.copy(tmp2)
        cv2 = np.copy(tmp2)
        skewness2 = np.copy(tmp2)
        kurtosis2 = np.copy(tmp2)

        # Pre-allocate 3D
        tmp3 = np.full(nt, np.nan)
        n3 = np.copy(tmp3)
        mean3 = np.copy(tmp3)
        median3 = np.copy(tmp3)
        min3 = np.copy(tmp3)
        max3 = np.copy(tmp3)
        std3 = np.copy(tmp3)
        cv3 = np.copy(tmp3)
        skewness3 = np.copy(tmp3)
        kurtosis3 = np.copy(tmp3)

        # Pre-allocate 4D
        tmp4 = np.full(1, np.nan)
        n4 = np.copy(tmp4)
        mean4 = np.copy(tmp4)
        median4 = np.copy(tmp4)
        min4 = np.copy(tmp4)
        max4 = np.copy(tmp4)
        std4 = np.copy(tmp4)
        cv4 = np.copy(tmp4)
        skewness4 = np.copy(tmp4)
        kurtosis4 = np.copy(tmp4)

        # Calculate statistics of the 4D volume
        intensities = image[roi]
        array_stats4 = FlatStats(intensities).calculate()
        n4 = array_stats4.n
        mean4 = array_stats4.mean
        median4 = array_stats4.median
        min4 = array_stats4.min
        max4 = array_stats4.max
        std4 = array_stats4.std
        cv4 = array_stats4.cv
        skewness4 = array_stats4.skewness
        kurtosis4 = array_stats4.kurtosis

        for it in range(nt):
            # Calculate statistics of each 3D volume
            it_intensities = image[:, :, :, it][roi[:, :, :, it]]
            array_stats3 = FlatStats(it_intensities).calculate()
            n3[it] = array_stats3.n
            mean3[it] = array_stats3.mean
            median3[it] = array_stats3.median
            min3[it] = array_stats3.min
            max3[it] = array_stats3.max
            std3[it] = array_stats3.std
            cv3[it] = array_stats3.cv
            skewness3[it] = array_stats3.skewness
            kurtosis3[it] = array_stats3.kurtosis

            for iz in range(nz):
                # Calculate statistics of each 2D slice
                iz_intensities = image[:, :, iz, it][roi[:, :, iz, it]]
                array_stats2 = FlatStats(iz_intensities).calculate()
                n2[iz, it] = array_stats2.n
                mean2[iz, it] = array_stats2.mean
                median2[iz, it] = array_stats2.median
                min2[iz, it] = array_stats2.min
                max2[iz, it] = array_stats2.max
                std2[iz, it] = array_stats2.std
                cv2[iz, it] = array_stats2.cv
                skewness2[iz, it] = array_stats2.skewness
                kurtosis2[iz, it] = array_stats2.kurtosis

        # Ensure statistics are None if data does not support calculation across
        # corresponding number of dimensions
        if self.image_ndims < 3:
            n3 = None
            mean3 = None
            median3 = None
            min3 = None
            max3 = None
            std3 = None
            cv3 = None
            skewness3 = None
            kurtosis3 = None
        if self.image_ndims < 4:
            n4 = None
            mean4 = None
            median4 = None
            min4 = None
            max4 = None
            std4 = None
            cv4 = None
            skewness4 = None
            kurtosis4 = None

        # Init dict for each dimension of each statistic
        n = {
            '2D': n2,  # number of voxels in each 2D slice
            '3D': n3,  # number of voxels in each 3D volume
            '4D': n4   # number of voxels in 4D volume
        }
        mean = {
            '2D': mean2,  # mean of each 2D slice
            '3D': mean3,  # mean of each 3D volume
            '4D': mean4   # mean of the 4D volume
        }
        median = {
            '2D': median2,
            '3D': median3,
            '4D': median4
        }
        minimum = {
            '2D': min2,
            '3D': min3,
            '4D': min4
        }
        maximum = {
            '2D': max2,
            '3D': max3,
            '4D': max4
        }
        std = {
            '2D': std2,
            '3D': std3,
            '4D': std4
        }
        cv = {
            '2D': cv2,
            '3D': cv3,
            '4D': cv4
        }
        skewness = {
            '2D': skewness2,
            '3D': skewness3,
            '4D': skewness4
        }
        kurtosis = {
            '2D': kurtosis2,
            '3D': kurtosis3,
            '4D': kurtosis4
        }

        # Init statistics "wrapper" dictionary
        statistics = {
            'n': n,
            'mean': mean,
            'median': median,
            'min': minimum,
            'max': maximum,
            'std': std,
            'cv': cv,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

        return statistics


class FlatStats():
    """Helper class that calculates statistics of a flat (1D) array

    Parameters
    ----------
    x : np.ndarray
        flat array (1 dimension)
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        See the documentation of stats.skew() or stats.kurtosis()

    Attributes
    ----------
    x : see above (parameters)
    nan_policy : see above (parameters)
    n : float
        number of array elements
    mean : float
    median : float
    min : float
    max : float
    std : float
        standard deviation
    cv : float
        coefficient of variation (std/mean)
    skewness : float
    kurtosis : float

    """
    def __init__(self, x, nan_policy='raise'):
        """ Init method: see class documentation for parameters/attributes

        """
        if np.ndim(x) != 1:
            raise ValueError("`x` should be a flat (1D) array")

        if np.isnan(x).any():
            raise ValueError("`x` must not contain nans")

        self.x = x
        self.nan_policy = nan_policy

        # Init statistics (calculated in calculate())
        self.n = NOT_CALCULATED_MSG
        self.mean = NOT_CALCULATED_MSG
        self.median = NOT_CALCULATED_MSG
        self.min = NOT_CALCULATED_MSG
        self.max = NOT_CALCULATED_MSG
        self.std = NOT_CALCULATED_MSG
        self.cv = NOT_CALCULATED_MSG
        self.skewness = NOT_CALCULATED_MSG
        self.kurtosis = NOT_CALCULATED_MSG

    def calculate(self):
        """Calculate flat array statistics

        Returns
        -------
        FlatStats object with calculated statistics

        """
        x = self.x
        nan_policy = self.nan_policy

        # Calculate statistics
        if x is None or x.size == 0:
            n = 0
            mean = np.nan
            median = np.nan
            minimum = np.nan
            maximum = np.nan
            std = np.nan
            cv = np.nan
            skewness = np.nan
            kurtosis = np.nan
        elif np.isnan(x).any():
            n = np.nan
            mean = np.nan
            median = np.nan
            minimum = np.nan
            maximum = np.nan
            std = np.nan
            cv = np.nan
            skewness = np.nan
            kurtosis = np.nan
        else:
            n = len(x)
            mean = np.mean(x)
            median = np.median(x)
            minimum = np.min(x)
            maximum = np.max(x)
            std = np.std(x)
            cv = std/mean
            skewness = stats.skew(x, bias=True, nan_policy=nan_policy)
            kurtosis = stats.kurtosis(x, fisher=True, bias=True, nan_policy=nan_policy)

        # Save in object
        self.n = n
        self.mean = mean
        self.median = median
        self.min = minimum
        self.max = maximum
        self.std = std
        self.cv = cv
        self.skewness = skewness
        self.kurtosis = kurtosis

        return self
