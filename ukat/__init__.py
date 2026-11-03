import sys

if sys.platform == 'win32':
    # On Windows the first msvcp140.dll loaded into the process is reused by
    # every other DLL. The scikit-learn 1.2 wheels bundle an old copy
    # (14.32) that is too old for ITK (built with 14.4x) and makes
    # itk.MultiThreaderBase.New() crash with an access violation if sklearn
    # is imported first. ITK loads its native libraries lazily, so force that
    # to happen before anything imports sklearn.
    try:
        import itk
        itk.MultiThreaderBase.New()
    except ImportError:
        pass

from . import data, mapping, qa, segmentation, utils, vessels
