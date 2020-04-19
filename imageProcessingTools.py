import numpy as np
import pydicom
from skimage.transform import resize
from skimage.restoration import unwrap_phase

# This is a script containing different methods that are auxiliary in 
# Image Analysis and might be used by multiple algorithms.

def unWrapPhase(pixelArray):
    return unwrap_phase(pixelArray)


def invertPixelArray(pixelArray):
    return np.invert(pixelArray)


def squarePixelArray(pixelArray):
    return np.square(pixelArray)

# I wrote these 3 very simple methods to demonstrate what this file should look like.
# Feel free to include more complex methods that involve thresholding, filtering, etc.

def resizePixelArray(pixelArray, pixelSpacing, reconstPixel = None):
    """Resizes the given pixelArray, using reconstPixel as reference of the resizing.
		This method assumes that the dimension axes=0 of pixelArray is the number of slices for np.shape > 2.""" 
    # This is so that data shares the same resolution and size

    if reconstPixel is not None:
        fraction = reconstPixel / pixelSpacing
    else:
        fraction = 1
		
    if len(np.shape(pixelArray)) == 2:
		pixelArray = resize(pixelArray, (pixelArray.shape[0] // fraction, pixelArray.shape[1] // fraction), anti_aliasing=True)
    elif len(np.shape(pixelArray)) == 3:
        pixelArray = resize(pixelArray, (pixelArray.shape[0], pixelArray.shape[1] // fraction, pixelArray.shape[2] // fraction), anti_aliasing=True)
    elif len(np.shape(pixelArray)) == 4:
        pixelArray = resize(pixelArray, (pixelArray.shape[0], pixelArray.shape[1] , pixelArray.shape[2] // fraction, pixelArray.shape[3] // fraction), anti_aliasing=True)
    elif len(np.shape(pixelArray)) == 5:
        pixelArray = resize(pixelArray, (pixelArray.shape[0], pixelArray.shape[1], pixelArray.shape[2], pixelArray.shape[3] // fraction, pixelArray.shape[4] // fraction), anti_aliasing=True)
    elif len(np.shape(pixelArray)) == 6:
        pixelArray = resize(pixelArray, (pixelArray.shape[0], pixelArray.shape[1], pixelArray.shape[2], pixelArray.shape[3], pixelArray.shape[4] // fraction, pixelArray.shape[5] // fraction), anti_aliasing=True)

    return pixelArray