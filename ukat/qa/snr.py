import numpy as np

from sklearn.mixture import BayesianGaussianMixture


class Isnr:
    """
        Attributes
        ----------
        isnr : float
            Image signal to noise ratio.
        noise_mask : np.ndarray
            Mask of the ROI used to measure noise levels. Should be voxels
            outside the body.
        """

    def __init__(self, pixel_array, noise_mask=None):
        """Initialise a image signal to noise ratio (iSNR) class instance.
        Parameters
        ----------
        pixel_array : np.ndarray
            Array of voxels over which iSNR should be calculated.
        noise_mask : np.ndarray, optional
            A binary voxel mask where voxels representing background i.e.
            outside the body, are True. If no mask is supplied, one is
            estimated using a Bayesian Gaussian mixture model to segment
            background voxels.
        """
        self.pixel_array = pixel_array
        self.shape = pixel_array.shape
        self.dimensions = len(pixel_array.shape)

        if noise_mask is None:
            self.__mask_background__()
        else:
            self.noise_mask = noise_mask

        self.isnr = np.nan
        self.__snr__()

    def __mask_background__(self):
        gmm = BayesianGaussianMixture(n_components=3,
                                      random_state=0,
                                      max_iter=500)
        fit_prop = (128 ** 2) / np.prod(self.shape)
        gmm.fit(self.pixel_array.reshape(-1, 1)[::int(1//fit_prop)])
        clusters = gmm.predict(self.pixel_array.reshape(-1, 1)).reshape(
            self.shape)
        bg_label = np.argmin(gmm.means_)
        self.noise_mask = clusters == bg_label

    def __snr__(self):
        noise = np.std(self.pixel_array[self.noise_mask])
        signal = np.mean(self.pixel_array[~self.noise_mask])
        self.isnr = (signal/noise) * np.sqrt(2-(np.pi/2))
