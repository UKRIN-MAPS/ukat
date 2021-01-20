import pytest
import scipy
import numpy as np
import numpy.testing as npt
import ukat.utils.arraystats as ast


def calc_stats(x):
    """
    Gold standard (gs) implementation of calculation of statistical measures.

    Parameters
    ----------
    x : np.ndarray with 1 dimension

    Notes
    -----
    This is defined at the module level as it is used by both test Classes.

    """
    n = len(x)
    mean = np.mean(x)
    median = np.median(x)
    minimum = np.min(x)
    maximum = np.max(x)
    std = np.std(x)
    if mean == 0:
        cv = np.nan
    else:
        cv = std/mean
    skewness = scipy.stats.skew(x, bias=True)
    kurtosis = scipy.stats.kurtosis(x, fisher=True, bias=True)
    entropy = scipy.stats.entropy(x)

    return [n, mean, median, minimum, maximum, std, cv, skewness, kurtosis,
            entropy]


class TestFlatStats:

    def test_array_with_one_element(self):
        # Note `gs`: gold standard

        # Int
        x1 = np.array([5])
        gs_1 = calc_stats(x1)
        stats_1 = self.flatstats_to_array(ast.FlatStats(x1).calculate())
        npt.assert_allclose(stats_1, gs_1, rtol=1e-6)

        # Float
        x2 = np.array([5.3])
        gs_2 = calc_stats(x2)
        stats_2 = self.flatstats_to_array(ast.FlatStats(x2).calculate())
        npt.assert_allclose(stats_2, gs_2, rtol=1e-6)

        # Negative number
        x3 = np.array([-5.3])
        gs_3 = calc_stats(x3)
        stats_3 = self.flatstats_to_array(ast.FlatStats(x3).calculate())
        npt.assert_allclose(stats_3, gs_3, rtol=1e-6)

        # Zero
        x4 = np.array([0])
        gs_4 = calc_stats(x4)
        stats_4 = self.flatstats_to_array(ast.FlatStats(x4).calculate())
        npt.assert_allclose(stats_4, gs_4, rtol=1e-6, equal_nan=True)

        # Nan
        x5 = np.array([np.nan])
        gs_5 = calc_stats(x5)
        stats_5 = self.flatstats_to_array(ast.FlatStats(x5).calculate())
        npt.assert_allclose(stats_5, gs_5, rtol=1e-6, equal_nan=True)

    def test_array_with_multiple_elements(self):
        # Ints
        x1 = np.array([5, 9, 7, 12, 19])
        gs_1 = calc_stats(x1)
        stats_1 = self.flatstats_to_array(ast.FlatStats(x1).calculate())
        npt.assert_allclose(stats_1, gs_1, rtol=1e-6)

        # Float
        x2 = np.array([5, 9, np.sqrt(7), 12, 19])
        gs_2 = calc_stats(x2)
        stats_2 = self.flatstats_to_array(ast.FlatStats(x2).calculate())
        npt.assert_allclose(stats_2, gs_2, rtol=1e-6)

        # Negative numbers
        x3 = np.array([5, 9, -np.sqrt(7), -12, 19])
        gs_3 = calc_stats(x3)
        stats_3 = self.flatstats_to_array(ast.FlatStats(x3).calculate())
        npt.assert_allclose(stats_3, gs_3, rtol=1e-6)

        # Zero
        x4 = np.array([5, 0, np.sqrt(7), -12, 19])
        gs_4 = calc_stats(x4)
        stats_4 = self.flatstats_to_array(ast.FlatStats(x4).calculate())
        npt.assert_allclose(stats_4, gs_4, rtol=1e-6, equal_nan=True)

        # Nan
        x5 = np.array([5, 0, np.sqrt(7), -12, np.nan])
        gs_5 = calc_stats(x5)
        stats_5 = self.flatstats_to_array(ast.FlatStats(x5).calculate())
        npt.assert_allclose(stats_5, gs_5, rtol=1e-6, equal_nan=True)

    def test_empty_array(self):
        expected_stats = np.array([0,        # n
                                   np.nan,   # mean
                                   np.nan,   # median
                                   np.nan,   # min
                                   np.nan,   # max
                                   np.nan,   # std
                                   np.nan,   # cv
                                   np.nan,   # skewness
                                   np.nan,   # kurtosis
                                   np.nan])  # entropy

        x1 = np.array([])
        stats_1 = self.flatstats_to_array(ast.FlatStats(x1).calculate())
        npt.assert_allclose(stats_1, expected_stats, rtol=1e-6, equal_nan=True)

    def test_nonflat_array(self):
        x1 = None
        with pytest.raises(ValueError):
            ast.FlatStats(x1)

        x2 = np.array([[5, 9], [3, 4]])
        with pytest.raises(ValueError):
            ast.FlatStats(x2)

    def flatstats_to_array(self, fs):
        n = fs.n
        mean = fs.mean
        median = fs.median
        minimum = fs.min
        maximum = fs.max
        std = fs.std
        cv = fs.cv
        skewness = fs.skewness
        kurtosis = fs.kurtosis
        entropy = fs.entropy

        return [n, mean, median, minimum, maximum, std, cv, skewness, kurtosis,
                entropy]


class TestArrayStats:

    # Good arrays
    array_4D = np.nan*np.ones((3, 3, 2, 2))
    array_4D[:, :, 0, 0] = np.array([[4, 8, -1.2], [-4.8, 4, 2], [-3, 0, 5.0]])
    array_4D[:, :, 0, 1] = np.array([[3, 6, -3.7], [-3.1, 2, 0], [-4, 4, 4.3]])
    array_4D[:, :, 1, 0] = np.array([[2, 4, -2.2], [-1.7, 3, 3], [-2, 3, 3.2]])
    array_4D[:, :, 1, 1] = np.array([[2, 2, -3.8], [-0.9, 4, 5], [-0, 1, 5.0]])

    array_3D = np.nan*np.ones((3, 3, 2))
    array_3D[:, :, 0] = np.array([[3, 6, -3.7], [-3.1, 2, 0], [-4, 4, 4.3]])
    array_3D[:, :, 1] = np.array([[2, 4, -2.2], [-1.7, 3, 3], [-2, 3, 3.2]])

    array_2D = np.array([[3, 6, -3.7], [-3.1, 2, 0], [-4, 4, 4.3]])

    # Bad arrays
    array_5D = np.ones((2, 2, 2, 2, 2))
    array_1D = np.array([1])

    def test_4D_without_roi(self):
        # Calculate stats using ArrayStats
        stats = ast.ArrayStats(self.array_4D).calculate()

        # Calculate "gold standard" (gs) stats
        gs_4D = calc_stats(self.array_4D.flatten())
        gs_3D_0 = calc_stats(self.array_4D[:, :, :, 0].flatten())
        gs_3D_1 = calc_stats(self.array_4D[:, :, :, 1].flatten())
        gs_2D_0 = calc_stats(self.array_4D[:, :, 0, 0].flatten())
        gs_2D_1 = calc_stats(self.array_4D[:, :, 0, 1].flatten())
        gs_2D_2 = calc_stats(self.array_4D[:, :, 1, 0].flatten())
        gs_2D_3 = calc_stats(self.array_4D[:, :, 1, 1].flatten())

        gs_list = [gs_4D, gs_3D_0, gs_3D_1, gs_2D_0, gs_2D_1, gs_2D_2, gs_2D_3]

        # Perform tests without ROI
        self.group_tests_4D(stats, gs_list)

    def test_4D_with_roi(self):
        # "Random" ROI
        roi_4D = np.ones((3, 3, 2, 2), dtype=bool)
        roi_4D[:, :, 0, 0] = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        roi_4D[:, :, 0, 1] = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        roi_4D[:, :, 1, 0] = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
        roi_4D[:, :, 1, 1] = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]])

        # Calculate stats using ArrayStats
        stats_roi = ast.ArrayStats(self.array_4D, roi=roi_4D).calculate()

        # Calculate "gold standard" (gs) stats
        gs_4D_roi = calc_stats(self.array_4D[roi_4D])
        gs_3D_roi_0 = calc_stats(self.array_4D[:, :, :, 0][roi_4D[:, :, :, 0]])
        gs_3D_roi_1 = calc_stats(self.array_4D[:, :, :, 1][roi_4D[:, :, :, 1]])
        gs_2D_roi_0 = calc_stats(self.array_4D[:, :, 0, 0][roi_4D[:, :, 0, 0]])
        gs_2D_roi_1 = calc_stats(self.array_4D[:, :, 0, 1][roi_4D[:, :, 0, 1]])
        gs_2D_roi_2 = calc_stats(self.array_4D[:, :, 1, 0][roi_4D[:, :, 1, 0]])
        gs_2D_roi_3 = calc_stats(self.array_4D[:, :, 1, 1][roi_4D[:, :, 1, 1]])

        gs_list_roi = [gs_4D_roi, gs_3D_roi_0, gs_3D_roi_1,
                       gs_2D_roi_0, gs_2D_roi_1, gs_2D_roi_2, gs_2D_roi_3]

        # Perform tests with ROI
        self.group_tests_4D(stats_roi, gs_list_roi)

    def group_tests_4D(self, stats, gs_list):
        """
        Perform tests that compare outputs of ArrayStats.calculate() and the
        calc_stats() function above.

        Notes
        -----
        This is used by tests with and without ROI so it was abstracted in its
        own function
        """

        # Unpack list
        [gs_4D, gs_3D_0, gs_3D_1, gs_2D_0, gs_2D_1, gs_2D_2, gs_2D_3] = gs_list

        # Group gs stats to compare with output from ArrayStats
        gs_n_4D = gs_4D[0]
        gs_mean_4D = gs_4D[1]
        gs_median_4D = gs_4D[2]
        gs_minimum_4D = gs_4D[3]
        gs_maximum_4D = gs_4D[4]
        gs_std_4D = gs_4D[5]
        gs_cv_4D = gs_4D[6]
        gs_skewness_4D = gs_4D[7]
        gs_kurtosis_4D = gs_4D[8]
        gs_entropy_4D = gs_4D[9]

        gs_n_3D = np.array([gs_3D_0[0], gs_3D_1[0]])
        gs_mean_3D = np.array([gs_3D_0[1], gs_3D_1[1]])
        gs_median_3D = np.array([gs_3D_0[2], gs_3D_1[2]])
        gs_minimum_3D = np.array([gs_3D_0[3], gs_3D_1[3]])
        gs_maximum_3D = np.array([gs_3D_0[4], gs_3D_1[4]])
        gs_std_3D = np.array([gs_3D_0[5], gs_3D_1[5]])
        gs_cv_3D = np.array([gs_3D_0[6], gs_3D_1[6]])
        gs_skewness_3D = np.array([gs_3D_0[7], gs_3D_1[7]])
        gs_kurtosis_3D = np.array([gs_3D_0[8], gs_3D_1[8]])
        gs_entropy_3D = np.array([gs_3D_0[9], gs_3D_1[9]])

        gs_n_2D = np.array([[gs_2D_0[0], gs_2D_1[0]],
                            [gs_2D_2[0], gs_2D_3[0]]])
        gs_mean_2D = np.array([[gs_2D_0[1], gs_2D_1[1]],
                               [gs_2D_2[1], gs_2D_3[1]]])
        gs_median_2D = np.array([[gs_2D_0[2], gs_2D_1[2]],
                                 [gs_2D_2[2], gs_2D_3[2]]])
        gs_minimum_2D = np.array([[gs_2D_0[3], gs_2D_1[3]],
                                  [gs_2D_2[3], gs_2D_3[3]]])
        gs_maximum_2D = np.array([[gs_2D_0[4], gs_2D_1[4]],
                                  [gs_2D_2[4], gs_2D_3[4]]])
        gs_std_2D = np.array([[gs_2D_0[5], gs_2D_1[5]],
                              [gs_2D_2[5], gs_2D_3[5]]])
        gs_cv_2D = np.array([[gs_2D_0[6], gs_2D_1[6]],
                             [gs_2D_2[6], gs_2D_3[6]]])
        gs_skewness_2D = np.array([[gs_2D_0[7], gs_2D_1[7]],
                                   [gs_2D_2[7], gs_2D_3[7]]])
        gs_kurtosis_2D = np.array([[gs_2D_0[8], gs_2D_1[8]],
                                   [gs_2D_2[8], gs_2D_3[8]]])
        gs_entropy_2D = np.array([[gs_2D_0[9], gs_2D_1[9]],
                                  [gs_2D_2[9], gs_2D_3[9]]])

        # Tests
        npt.assert_allclose(stats["n"]["4D"], gs_n_4D, rtol=1e-6)
        npt.assert_allclose(stats["mean"]["4D"], gs_mean_4D, rtol=1e-6)
        npt.assert_allclose(stats["median"]["4D"], gs_median_4D, rtol=1e-6)
        npt.assert_allclose(stats["min"]["4D"], gs_minimum_4D, rtol=1e-6)
        npt.assert_allclose(stats["max"]["4D"], gs_maximum_4D, rtol=1e-6)
        npt.assert_allclose(stats["std"]["4D"], gs_std_4D, rtol=1e-6)
        npt.assert_allclose(stats["cv"]["4D"], gs_cv_4D, rtol=1e-6)
        npt.assert_allclose(stats["skewness"]["4D"], gs_skewness_4D, rtol=1e-6)
        npt.assert_allclose(stats["kurtosis"]["4D"], gs_kurtosis_4D, rtol=1e-6)
        npt.assert_allclose(stats["entropy"]["4D"], gs_entropy_4D, rtol=1e-6)

        npt.assert_allclose(stats["n"]["3D"], gs_n_3D, rtol=1e-6)
        npt.assert_allclose(stats["mean"]["3D"], gs_mean_3D, rtol=1e-6)
        npt.assert_allclose(stats["median"]["3D"], gs_median_3D, rtol=1e-6)
        npt.assert_allclose(stats["min"]["3D"], gs_minimum_3D, rtol=1e-6)
        npt.assert_allclose(stats["max"]["3D"], gs_maximum_3D, rtol=1e-6)
        npt.assert_allclose(stats["std"]["3D"], gs_std_3D, rtol=1e-6)
        npt.assert_allclose(stats["cv"]["3D"], gs_cv_3D, rtol=1e-6)
        npt.assert_allclose(stats["skewness"]["3D"], gs_skewness_3D, rtol=1e-6)
        npt.assert_allclose(stats["kurtosis"]["3D"], gs_kurtosis_3D, rtol=1e-6)
        npt.assert_allclose(stats["entropy"]["3D"], gs_entropy_3D, rtol=1e-6)

        npt.assert_allclose(stats["n"]["2D"], gs_n_2D, rtol=1e-6)
        npt.assert_allclose(stats["mean"]["2D"], gs_mean_2D, rtol=1e-6)
        npt.assert_allclose(stats["median"]["2D"], gs_median_2D, rtol=1e-6)
        npt.assert_allclose(stats["min"]["2D"], gs_minimum_2D, rtol=1e-6)
        npt.assert_allclose(stats["max"]["2D"], gs_maximum_2D, rtol=1e-6)
        npt.assert_allclose(stats["std"]["2D"], gs_std_2D, rtol=1e-6)
        npt.assert_allclose(stats["cv"]["2D"], gs_cv_2D, rtol=1e-6)
        npt.assert_allclose(stats["skewness"]["2D"], gs_skewness_2D, rtol=1e-6)
        npt.assert_allclose(stats["kurtosis"]["2D"], gs_kurtosis_2D, rtol=1e-6)
        npt.assert_allclose(stats["entropy"]["2D"], gs_entropy_2D, rtol=1e-6)

    def test_3D_without_roi(self):
        # Calculate stats using ArrayStats
        stats = ast.ArrayStats(self.array_3D).calculate()

        # Calculate "gold standard" (gs) stats
        gs_3D = calc_stats(self.array_3D.flatten())
        gs_2D_0 = calc_stats(self.array_3D[:, :, 0].flatten())
        gs_2D_1 = calc_stats(self.array_3D[:, :, 1].flatten())

        # Group gs stats to compare with output from ArrayStats
        gs_n_3D = gs_3D[0]
        gs_mean_3D = gs_3D[1]
        gs_median_3D = gs_3D[2]
        gs_minimum_3D = gs_3D[3]
        gs_maximum_3D = gs_3D[4]
        gs_std_3D = gs_3D[5]
        gs_cv_3D = gs_3D[6]
        gs_skewness_3D = gs_3D[7]
        gs_kurtosis_3D = gs_3D[8]
        gs_entropy_3D = gs_3D[9]

        gs_n_2D = np.array([gs_2D_0[0], gs_2D_1[0]])
        gs_mean_2D = np.array([gs_2D_0[1], gs_2D_1[1]])
        gs_median_2D = np.array([gs_2D_0[2], gs_2D_1[2]])
        gs_minimum_2D = np.array([gs_2D_0[3], gs_2D_1[3]])
        gs_maximum_2D = np.array([gs_2D_0[4], gs_2D_1[4]])
        gs_std_2D = np.array([gs_2D_0[5], gs_2D_1[5]])
        gs_cv_2D = np.array([gs_2D_0[6], gs_2D_1[6]])
        gs_skewness_2D = np.array([gs_2D_0[7], gs_2D_1[7]])
        gs_kurtosis_2D = np.array([gs_2D_0[8], gs_2D_1[8]])
        gs_entropy_2D = np.array([gs_2D_0[9], gs_2D_1[9]])

        npt.assert_allclose(stats["n"]["3D"], gs_n_3D, rtol=1e-6)
        npt.assert_allclose(stats["mean"]["3D"], gs_mean_3D, rtol=1e-6)
        npt.assert_allclose(stats["median"]["3D"], gs_median_3D, rtol=1e-6)
        npt.assert_allclose(stats["min"]["3D"], gs_minimum_3D, rtol=1e-6)
        npt.assert_allclose(stats["max"]["3D"], gs_maximum_3D, rtol=1e-6)
        npt.assert_allclose(stats["std"]["3D"], gs_std_3D, rtol=1e-6)
        npt.assert_allclose(stats["cv"]["3D"], gs_cv_3D, rtol=1e-6)
        npt.assert_allclose(stats["skewness"]["3D"], gs_skewness_3D, rtol=1e-6)
        npt.assert_allclose(stats["kurtosis"]["3D"], gs_kurtosis_3D, rtol=1e-6)
        npt.assert_allclose(stats["entropy"]["3D"], gs_entropy_3D, rtol=1e-6)

        npt.assert_allclose(stats["n"]["2D"], gs_n_2D, rtol=1e-6)
        npt.assert_allclose(stats["mean"]["2D"], gs_mean_2D, rtol=1e-6)
        npt.assert_allclose(stats["median"]["2D"], gs_median_2D, rtol=1e-6)
        npt.assert_allclose(stats["min"]["2D"], gs_minimum_2D, rtol=1e-6)
        npt.assert_allclose(stats["max"]["2D"], gs_maximum_2D, rtol=1e-6)
        npt.assert_allclose(stats["std"]["2D"], gs_std_2D, rtol=1e-6)
        npt.assert_allclose(stats["cv"]["2D"], gs_cv_2D, rtol=1e-6)
        npt.assert_allclose(stats["skewness"]["2D"], gs_skewness_2D, rtol=1e-6)
        npt.assert_allclose(stats["kurtosis"]["2D"], gs_kurtosis_2D, rtol=1e-6)
        npt.assert_allclose(stats["entropy"]["2D"], gs_entropy_2D, rtol=1e-6)

    def test_2D_without_roi(self):
        # Calculate stats using ArrayStats
        stats = ast.ArrayStats(self.array_2D).calculate()

        # Calculate "gold standard" (gs) stats
        gs_2D = calc_stats(self.array_2D.flatten())

        # Group gs stats to compare with output from ArrayStats
        gs_n_2D = gs_2D[0]
        gs_mean_2D = gs_2D[1]
        gs_median_2D = gs_2D[2]
        gs_minimum_2D = gs_2D[3]
        gs_maximum_2D = gs_2D[4]
        gs_std_2D = gs_2D[5]
        gs_cv_2D = gs_2D[6]
        gs_skewness_2D = gs_2D[7]
        gs_kurtosis_2D = gs_2D[8]
        gs_entropy_2D = gs_2D[9]

        npt.assert_allclose(stats["n"], gs_n_2D, rtol=1e-6)
        npt.assert_allclose(stats["mean"], gs_mean_2D, rtol=1e-6)
        npt.assert_allclose(stats["median"], gs_median_2D, rtol=1e-6)
        npt.assert_allclose(stats["min"], gs_minimum_2D, rtol=1e-6)
        npt.assert_allclose(stats["max"], gs_maximum_2D, rtol=1e-6)
        npt.assert_allclose(stats["std"], gs_std_2D, rtol=1e-6)
        npt.assert_allclose(stats["cv"], gs_cv_2D, rtol=1e-6)
        npt.assert_allclose(stats["skewness"], gs_skewness_2D, rtol=1e-6)
        npt.assert_allclose(stats["kurtosis"], gs_kurtosis_2D, rtol=1e-6)
        npt.assert_allclose(stats["entropy"], gs_entropy_2D, rtol=1e-6)

    def test_bad_array_dimensions(self):
        with pytest.raises(ValueError):
            ast.ArrayStats(self.array_1D)

        with pytest.raises(ValueError):
            ast.ArrayStats(self.array_5D)

    def test_bad_roi_dtype(self):
        array = np.ones((3, 3, 2))
        roi = np.ones((3, 3, 2))
        with pytest.raises(TypeError):
            ast.ArrayStats(array, roi)

    def test_roi_shape_mismatch(self):
        array = np.ones((3, 3, 2))
        roi = np.ones((3, 3, 3), dtype=bool)
        with pytest.raises(ValueError):
            ast.ArrayStats(array, roi)
