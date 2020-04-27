import numpy as np
from imageProcessingTools import unWrapPhase


def B0Map(pixelArray, echoList):
    """Documentation to be written in the future"""
    try:
        if len(echoList) > 1:
            # Is the given array already a Phase Difference or not?
            phaseDiffOriginal = (np.squeeze(pixelArray[1, ...])
                                 - np.squeeze(pixelArray[0, ...]))
            deltaTE = np.absolute(echoList[1] - echoList[0]) * 0.001
            # Conversion from ms to s
        else:
            # If it's a Phase Difference, it just unwraps the phase
            derivedImage = unWrapPhase(pixelArray)
            return derivedImage
        # Normalise to -2Pi and +2Pi
        phaseDiff = (phaseDiffOriginal / ((1 / (2 * np.pi))
                     * np.amax(phaseDiffOriginal)
                     * np.ones(np.shape(phaseDiffOriginal))))
        derivedImage = (unWrapPhase(phaseDiff) / ((2 * np.pi * deltaTE)
                        * np.ones(np.shape(phaseDiff))))
        del phaseDiffOriginal, phaseDiff, deltaTE
        return derivedImage
    except Exception as e:
        print('Error in function B0MapCode.B0Map: ' + str(e))
