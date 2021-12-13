import nibabel as nib
import numpy as np

from skimage.measure import label, regionprops
from skimage.transform import resize
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from ukat.data import fetch


def rescale(data):
    black = np.mean(data) - 0.5 * np.std(data)
    if black < data.min():
        black = data.min()
    white = np.mean(data) + 4 * np.std(data)
    if white > data.max():
        white = data.max()
    data = np.clip(data, black, white) - black
    data = data / (white - black)
    return data


class Mask:

    def __init__(self, binary=True):
        self.binary = binary
        self.model = Model()

        # Initialise output attributes
        #TODO error thrown if fit not already called
        self.pixel_array = np.nan
        self.shape = self.pixel_array.shape
        self.affine = np.nan
        self.img = nib.Nifti1Image(self.pixel_array, self.affine)
        self.zoom = self.img.header.get_zooms()
        self.mask = np.zeros(self.shape)
        self.labels = np.zeros(self.shape)
        self.masked_pixel_array = self.pixel_array * self.mask
        self.tkv = np.nan
        self.lkv = np.nan
        self.rkv = np.nan

        self.mask = self.__generate_mask__()

    def fit(self, pixel_array, affine):
        self.pixel_array = pixel_array
        self.shape = self.pixel_array.shape
        self.affine = affine
        self.img = nib.Nifti1Image(self.pixel_array, self.affine)
        self.zoom = self.img.header.get_zooms()

    def __generate_mask__(self):
        pre_processed_data = self.__preprocess__(self.pixel_array)
        prediction = self.model.predict(pre_processed_data)
        mask_raw = self.__inverse_preprocess(prediction)
        clean_mask = self.__clean_mask__(mask_raw)
        return clean_mask

    @staticmethod
    def __preprocess__(pixel_array):
        pixel_array = np.flip(pixel_array, 1)
        pixel_array = np.swapaxes(pixel_array, 0, 2)
        pixel_array = np.swapaxes(pixel_array, 1, 2)
        pixel_array = rescale(pixel_array)
        pixel_array = resize(pixel_array, (pixel_array.shape[0], 256, 256))
        pixel_array = pixel_array.reshape((*pixel_array.shape, 1))
        return pixel_array

    @staticmethod
    def __inverse_preprocess(self):
        pixel_array = self.pixel_array  # Placeholder
        return pixel_array

    @staticmethod
    def __clean_mask__(mask_raw):
        labels = label(mask_raw, connectivity=1)
        props = regionprops(labels)
        areas = [region.area for region in props]
        kidney_labels = np.argpartition(areas, -2)[-2:]
        clean_mask = np.zeros(mask_raw.shape)
        clean_mask[labels == props[kidney_labels[0]].label] = 1
        clean_mask[labels == props[kidney_labels[1]].label] = 1
        return clean_mask


class Model:

    def __init__(self, weights=None):
        if weights is None:
            weights = fetch.get_fnames('total_kidney_weights')[0]
        self.model = load_model(weights,
                                custom_objects={'dice_coef_loss':
                                                self.__dice_coef_loss__,
                                                'dice_coef':
                                                self.__dice_coef__})

    def predict(self, data, batch_size=2 ** 3):
        return self.model.predict(data, batch_size=batch_size)

    @staticmethod
    def __dice_coef__(true, pred):
        smooth = 1
        true_f = K.flatten(true)
        pred_f = K.flatten(pred)
        intersection = K.sum(true_f * pred_f)
        return (2. * intersection + smooth) / (
                    K.sum(true_f) + K.sum(pred_f) + smooth)

    def __dice_coef_loss__(self, true, pred):
        loss = 1 - self.__dice_coef__(true, pred)
        return loss


class Segmentation(nib.Nifti1Image):
    def __init__(self, pixel_array, affine, mask):
        super().__init__(pixel_array, affine)
        self._mask = mask

    def get_mask(self):
        return self._mask

    def tkv(self):
        return np.sum(self._mask) * np.prod(self.header.get_zooms())
