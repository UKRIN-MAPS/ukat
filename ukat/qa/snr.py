import nibabel as nib
import numpy as np
import os

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
        clusters : np.ndarray
            The labels assigned to each voxel during automatic background
            segmentation. This can be useful for debugging.
        """

    def __init__(self, pixel_array, noise_mask=None, n_clusters=3):
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
        n_clusters : int, optional
            When using the automatic background segmentation this is the
            total number of componenets the image is segmented into. The
            component with the lowest mean is assumed to be background.
            Default 3.
        """
        self.pixel_array = pixel_array
        self.n_clusters = n_clusters
        self.shape = pixel_array.shape
        self.dimensions = len(pixel_array.shape)

        if noise_mask is None:
            self.__mask_background__()
        else:
            self.noise_mask = noise_mask
            self.clusters = np.nan

        self.isnr = np.nan
        self.__snr__()

    def __mask_background__(self):
        np.random.seed(0)
        gmm = BayesianGaussianMixture(n_components=self.n_clusters,
                                      random_state=0,
                                      max_iter=500)
        # Because gmm's can get quite slow when fitting to large images,
        # we randomly sample a number of voxels equivalent to a 128 x 128 x
        # 3 image to keep runtimes consistent and manageable for large
        # multi-te/ti/dynamic images.
        fit_prop = (128 ** 2 * 3) / np.prod(self.shape)
        fit_mask = np.random.rand(self.pixel_array.size) < fit_prop
        gmm.fit(self.pixel_array.reshape(-1, 1)[fit_mask])
        self.clusters = gmm.predict(self.pixel_array.reshape(-1, 1)).reshape(
            self.shape)
        bg_label = np.argmin(gmm.means_)
        self.noise_mask = self.clusters == bg_label

    def __snr__(self):
        noise = np.std(self.pixel_array[self.noise_mask])
        signal = np.mean(self.pixel_array[~self.noise_mask])
        self.isnr = (signal/noise) * np.sqrt(2-(np.pi/2))


class Tsnr:
    """
    Attributes
    ----------
    tsnr_map : np.ndarray
        Map of temporal signal to noise ratio.
    """

    def __init__(self, pixel_array, affine, mask=None):
        """Initialise a temporal signal to noise ratio (tSNR) class instance.

        Parameters
        ----------
        pixel_array : np.ndarray
            A array containing the signal from each voxel with the last
            dimension being repeated dynamics i.e. the array needed to
            generate a tSNR map would have dimensions [x, y, z, d].
        affine : np.ndarray
            A matrix giving the relationship between voxel coordinates and
            world coordinates.
        mask : np.ndarray, optional
            A boolean mask of the voxels to fit. Should be the shape of the
            desired tSNR map rather than the raw data i.e. omit the dynamics
            dimension.
        """
        np.seterr(divide='ignore', invalid='ignore')
        self.pixel_array = pixel_array
        self.shape = pixel_array.shape[:-1]
        self.dimensions = len(pixel_array.shape)
        self.n_d = pixel_array.shape[-1]
        self.affine = affine
        # Generate a mask if there isn't one specified
        if mask is None:
            self.mask = np.ones(self.shape, dtype=bool)
        else:
            self.mask = mask
        # Don't process any nan values
        self.mask[np.isnan(np.sum(pixel_array, axis=-1))] = False

        self.pixel_array = self.pixel_array * \
            np.repeat(self.mask[..., np.newaxis],
                      self.n_d, axis=-1)

        # Initialise output attributes
        self.tsnr_map = np.zeros(self.shape)
        self.tsnr_map = self.__tsnr__()

    def __tsnr__(self):
        # Regress out linear and quadratic temporal drifts associated with
        # hardware using a GLM of the form Y = X * beta + error
        # as in Hutton C et al. The impact of physiological noise correction
        # on fMRI at 7T. NeuroImage 2011;57:101–112 doi:
        # 10.1016/j.neuroimage.2011.04.018.

        # Vectorise image
        pixel_array_vector = np.reshape(self.pixel_array,
                                        (np.prod(self.shape), self.n_d))
        x = np.vstack([np.ones(self.n_d),
                       np.arange(1, self.n_d + 1),
                       np.arange(1, self.n_d + 1) ** 2]).T
        beta = np.linalg.pinv(x).dot(pixel_array_vector.T)
        pixel_array_vector_detrended = pixel_array_vector.T - \
            x[:, 1:].dot(beta[1:])
        pixel_array_detrended = pixel_array_vector_detrended.T.reshape((
            *self.shape, self.n_d))
        tsnr_map = pixel_array_detrended.mean(axis=-1) / \
            pixel_array_detrended.std(axis=-1)  # Might want to try
        # ddof=1 as per Kevins code...
        tsnr_map[tsnr_map > 1000] = 0
        tsnr_map = np.nan_to_num(tsnr_map)
        return tsnr_map

    def to_nifti(self, output_directory=os.getcwd(), base_file_name='Output'):
        """Exports tSNR maps to NIFTI.

        Parameters
        ----------
        output_directory : string, optional
            Path to the folder where the NIFTI files will be saved.
        base_file_name : string, optional
            Filename of the resulting NIFTI. This code appends the extension.
            Eg., base_file_name = 'Output' will result in 'Output.nii.gz'.
        """
        os.makedirs(output_directory, exist_ok=True)
        base_path = os.path.join(output_directory, base_file_name)

        tsnr_nifti = nib.Nifti1Image(self.tsnr_map, affine=self.affine)
        nib.save(tsnr_nifti, base_path + '_tsnr_map.nii.gz')
