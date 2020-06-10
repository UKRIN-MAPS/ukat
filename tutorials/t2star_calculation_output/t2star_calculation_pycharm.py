# Ensure `data` and `methods` can be imported
import sys
sys.path.insert(0, '..')

# Other imports
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import warnings

# UKRIN-MAPS modules
import data.fetch                  # Test data fetcher
from methods.T2Star import T2Star  # T2* mapping module

# Enable inline plotting; hide T2Star_Nottingham() RuntimeWarnings
#%matplotlib inline
#warnings.filterwarnings('ignore')

# Initialise output path for T2star map
OUTPUT_DIR = os.path.join(os.getcwd(), "t2star_calculation_output")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'T2StarMap.nii.gz')

# Fetch test data
image, affine, TEs = data.fetch.r2star_ge()
