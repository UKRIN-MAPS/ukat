# Part of standard library or ukat
import os
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.restoration import unwrap_phase
from ukat.utils.tools import convert_to_pi_range

# fslpy modules (not part of ukat, need separate installation)
from fsl.wrappers.fugue import prelude
from fsl.wrappers import LOAD

# Modules that are not part of ukat but live in this experiments' folder
from arraystats import ArrayStats
from roi import RegionOfInterest

# Stuff for debugging (not needed, ask Fabio)
# from wip.vis import Formatter, pixelinfo

# CONSTANTS
DIR_ROOT = os.getcwd()
DIR_DATA = os.path.join(DIR_ROOT, "data")

# DATASETS correspond to the name of each subdir with each dataset
DATASETS = ['ge_01',
            'philips_01',
            'philips_02',
            'philips_03',
            'siemens_01',
            'philips_04',
            'philips_05',
            'philips_06',
            'philips_07',
            'philips_08',
            'philips_09',
            'philips_10']

# Don't change slices to show as some datasets only have ROIs drawn for this
# particular slice
SLICES_TO_SHOW = [8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

FILENAMES_ALL = []
FILENAMES_ALL.append(['00720__Magnitude_ims__B0_map_dual_echo_e1.nii.gz',
                      '00721__Phase_ims__B0_map_dual_echo_e1.nii.gz',
                      '00720__Magnitude_ims__B0_map_dual_echo_e2.nii.gz',
                      '00721__Phase_ims__B0_map_dual_echo_e2.nii.gz',
                      '00720__Magnitude_ims__B0_map_dual_echo_e1_roi-R.nii.gz',
                      '00720__Magnitude_ims__B0_map_dual_echo_e1_roi-L.nii.gz'])
FILENAMES_ALL.append(['01001__B0_map_expiration_volume_2DMS_product_auto_e1.nii.gz',
                      '01001__B0_map_expiration_volume_2DMS_product_auto_e1_ph.nii.gz',
                      '01001__B0_map_expiration_volume_2DMS_product_auto_e2.nii.gz',
                      '01001__B0_map_expiration_volume_2DMS_product_auto_e2_ph.nii.gz',
                      '01001__B0_map_expiration_volume_2DMS_product_auto_e1_roi-R.nii.gz',
                      '01001__B0_map_expiration_volume_2DMS_product_auto_e1_roi-L.nii.gz'])
FILENAMES_ALL.append(['01101__B0_map_expiration_volume_2DMS_product_auto_e1.nii.gz',
                      '01101__B0_map_expiration_volume_2DMS_product_auto_e1_ph.nii.gz',
                      '01101__B0_map_expiration_volume_2DMS_product_auto_e2.nii.gz',
                      '01101__B0_map_expiration_volume_2DMS_product_auto_e2_ph.nii.gz',
                      '01101__B0_map_expiration_volume_2DMS_product_auto_e1_roi-R.nii.gz',
                      '01101__B0_map_expiration_volume_2DMS_product_auto_e1_roi-L.nii.gz'])
FILENAMES_ALL.append(['01301__B0_map_expiration_volume_2DMS_product_auto_e1.nii.gz',
                      '01301__B0_map_expiration_volume_2DMS_product_auto_e1_ph.nii.gz',
                      '01301__B0_map_expiration_volume_2DMS_product_auto_e2.nii.gz',
                      '01301__B0_map_expiration_volume_2DMS_product_auto_e2_ph.nii.gz',
                      '01301__B0_map_expiration_volume_2DMS_product_auto_e1_roi-R.nii.gz',
                      '01301__B0_map_expiration_volume_2DMS_product_auto_e1_roi-L.nii.gz'])
FILENAMES_ALL.append(['0044_bh_b0map_fa3_default_bh_b0map_fa3_default_e1.nii.gz',
                      '0045_bh_b0map_fa3_default_bh_b0map_fa3_default_e1_ph.nii.gz',
                      '0044_bh_b0map_fa3_default_bh_b0map_fa3_default_e2.nii.gz',
                      '0045_bh_b0map_fa3_default_bh_b0map_fa3_default_e2_ph.nii.gz',
                      '0044_bh_b0map_fa3_default_bh_b0map_fa3_default_e1_roi-R.nii.gz',
                      '0044_bh_b0map_fa3_default_bh_b0map_fa3_default_e1_roi-L.nii.gz'])
FILENAMES_ALL.append(['00501__B0_map_expiration_volume_2DMS_delTE2.3_e1.nii.gz',
                      '00501__B0_map_expiration_volume_2DMS_delTE2.3_e1_ph.nii.gz',
                      '00501__B0_map_expiration_volume_2DMS_delTE2.3_e2.nii.gz',
                      '00501__B0_map_expiration_volume_2DMS_delTE2.3_e2_ph.nii.gz',
                      '00501__B0_map_expiration_volume_2DMS_delTE2.3_e1_roi-R.nii.gz',
                      '00501__B0_map_expiration_volume_2DMS_delTE2.3_e1_roi-L.nii.gz'])
FILENAMES_ALL.append(['00601__B0_map_expiration_volume_2DMS_delTE2.2_e1.nii.gz',
                      '00601__B0_map_expiration_volume_2DMS_delTE2.2_e1_ph.nii.gz',
                      '00601__B0_map_expiration_volume_2DMS_delTE2.2_e2.nii.gz',
                      '00601__B0_map_expiration_volume_2DMS_delTE2.2_e2_ph.nii.gz',
                      '00601__B0_map_expiration_volume_2DMS_delTE2.2_e1_roi-R.nii.gz',
                      '00601__B0_map_expiration_volume_2DMS_delTE2.2_e1_roi-L.nii.gz'])
FILENAMES_ALL.append(['00701__B0_map_expiration_volume_2DMS_delTE2.46_e1.nii.gz',
                      '00701__B0_map_expiration_volume_2DMS_delTE2.46_e1_ph.nii.gz',
                      '00701__B0_map_expiration_volume_2DMS_delTE2.46_e2.nii.gz',
                      '00701__B0_map_expiration_volume_2DMS_delTE2.46_e2_ph.nii.gz',
                      '00701__B0_map_expiration_volume_2DMS_delTE2.46_e1_roi-R.nii.gz',
                      '00701__B0_map_expiration_volume_2DMS_delTE2.46_e1_roi-L.nii.gz'])
FILENAMES_ALL.append(['00801__B0_map_expiration_volume_2DMS_delTE3_e1.nii.gz',
                      '00801__B0_map_expiration_volume_2DMS_delTE3_e1_ph.nii.gz',
                      '00801__B0_map_expiration_volume_2DMS_delTE3_e2.nii.gz',
                      '00801__B0_map_expiration_volume_2DMS_delTE3_e2_ph.nii.gz',
                      '00801__B0_map_expiration_volume_2DMS_delTE3_e1_roi-R.nii.gz',
                      '00801__B0_map_expiration_volume_2DMS_delTE3_e1_roi-L.nii.gz'])
FILENAMES_ALL.append(['00901__B0_map_expiration_volume_2DMS_delTE3.45_e1.nii.gz',
                      '00901__B0_map_expiration_volume_2DMS_delTE3.45_e1_ph.nii.gz',
                      '00901__B0_map_expiration_volume_2DMS_delTE3.45_e2.nii.gz',
                      '00901__B0_map_expiration_volume_2DMS_delTE3.45_e2_ph.nii.gz',
                      '00901__B0_map_expiration_volume_2DMS_delTE3.45_e1_roi-R.nii.gz',
                      '00901__B0_map_expiration_volume_2DMS_delTE3.45_e1_roi-L.nii.gz'])
FILENAMES_ALL.append(['01101__B0_map_expiration_volume_2DMS_delTEinphase_e1.nii.gz',
                      '01101__B0_map_expiration_volume_2DMS_delTEinphase_e1_ph.nii.gz',
                      '01101__B0_map_expiration_volume_2DMS_delTEinphase_e2.nii.gz',
                      '01101__B0_map_expiration_volume_2DMS_delTEinphase_e2_ph.nii.gz',
                      '01101__B0_map_expiration_volume_2DMS_delTEinphase_e1_roi-R.nii.gz',
                      '01101__B0_map_expiration_volume_2DMS_delTEinphase_e1_roi-L.nii.gz'])
FILENAMES_ALL.append(['01201__B0_map_expiration_volume_2DMS_delTEoutphase_e1.nii.gz',
                      '01201__B0_map_expiration_volume_2DMS_delTEoutphase_e1_ph.nii.gz',
                      '01201__B0_map_expiration_volume_2DMS_delTEoutphase_e2.nii.gz',
                      '01201__B0_map_expiration_volume_2DMS_delTEoutphase_e2_ph.nii.gz',
                      '01201__B0_map_expiration_volume_2DMS_delTEoutphase_e1_roi-R.nii.gz',
                      '01201__B0_map_expiration_volume_2DMS_delTEoutphase_e1_roi-L.nii.gz'])

# Define the main titles for the figure showing the unwrapping results for each dataset
SUPTITLES = []
SUPTITLES.append('ge_01 - 00720__Magnitude_ims__B0_map_dual_echo_e1: TEs (ms): [2.216, 5.136]')
SUPTITLES.append('philips_01 - 01001__B0_map_expiration_volume_2DMS_product_auto_e1 - default shim: TEs (ms): [4.001, 6.46]')
SUPTITLES.append('philips_02 - 01101__B0_map_expiration_volume_2DMS_product_auto_e1 - volume shim over kidneys: TEs (ms): [4.001, 6.46]')
SUPTITLES.append('philips_03 - 01301__B0_map_expiration_volume_2DMS_product_auto_e1 - volume shim over lungs: TEs (ms): [4.001, 6.46]')
SUPTITLES.append('siemens_01 - 0044_bh_b0map_fa3_default_bh_b0map_fa3_default_e1: TEs (ms): [4.000, 6.46]')
SUPTITLES.append('philips_04 - 00501__B0_map_expiration_volume_2DMS_delTE2.3: TEs (ms): [4.001, 6.3]')
SUPTITLES.append('philips_05 - 00601__B0_map_expiration_volume_2DMS_delTE2.2: TEs (ms): [4.001, 6.2]')
SUPTITLES.append('philips_06 - 00701__B0_map_expiration_volume_2DMS_delTE2.46: TEs (ms): [4.001, 6.46]')
SUPTITLES.append('philips_07 - 00801__B0_map_expiration_volume_2DMS_delTE3: TEs (ms): [4.001, 7.0]')
SUPTITLES.append('philips_08 - 00901__B0_map_expiration_volume_2DMS_delTE3.45: TEs (ms): [4.001, 7.45]')
SUPTITLES.append('philips_09 - 01101__B0_map_expiration_volume_2DMS_delTEinphase: TEs (ms): [4.001, 6.907]')
SUPTITLES.append('philips_10 - 01201__B0_map_expiration_volume_2DMS_delTEoutphase: TEs (ms): [4.001, 5.756]')

SLICE_SPECIFIC_INTENSITY_RANGES = False  # not implemented, has to be False


class CompareUnwrapping():
    """Experimental class to perform unwrapping with several methods and plot
    the results in a figure montage"""

    def __init__(self, dir_dataset, filenames):
        """Constructor for CompareUnwrapping class"""

        filepaths = [os.path.join(dir_dataset, filename) for filename in filenames]

        [self.magnitude_e1_path, self.phase_e1_path,
         self.magnitude_e2_path, self.phase_e2_path,
         self.roir_path, self.roil_path] = filepaths

        self.load_data()
        self.convert_to_rad()
        self.unwrap()

    def load_data(self):
        """Load NIfTI files into numpy arrays"""

        # Init both nib objects and np arrays (the former needed for prelude)

        # Magnitude
        magnitude_e1_nib = nib.load(self.magnitude_e1_path)
        magnitude_e2_nib = nib.load(self.magnitude_e2_path)
        magnitude_e1 = magnitude_e1_nib.get_fdata()
        magnitude_e2 = magnitude_e2_nib.get_fdata()

        self.magnitude_e1_nib = magnitude_e1_nib
        self.magnitude_e2_nib = magnitude_e2_nib
        self.magnitude_e1 = magnitude_e1
        self.magnitude_e2 = magnitude_e2

        # Phase
        phase_e1_nib = nib.load(self.phase_e1_path)
        phase_e2_nib = nib.load(self.phase_e2_path)
        phase_e1 = phase_e1_nib.get_fdata()
        phase_e2 = phase_e2_nib.get_fdata()

        self.phase_e1_nib = phase_e1_nib
        self.phase_e2_nib = phase_e2_nib
        self.phase_e1 = phase_e1
        self.phase_e2 = phase_e2

        roi = RegionOfInterest([self.roir_path, self.roil_path])
        roi.add()

        self.roib = roi.data[0]

        return self

    def convert_to_rad(self):
        """Convert phase images into radians and "save" them into new
        NIfTI1Image objects (necessary for prelude)"""

        # Convert to rad and save as nib object -------------------------------
        phase_e1_rad = convert_to_pi_range(self.phase_e1)
        phase_e2_rad = convert_to_pi_range(self.phase_e2)
        phase_e1_rad_nib = nib.Nifti1Image(phase_e1_rad, self.phase_e1_nib.affine)
        phase_e2_rad_nib = nib.Nifti1Image(phase_e2_rad, self.phase_e2_nib.affine)

        self.phase_e1_rad = phase_e1_rad
        self.phase_e2_rad = phase_e2_rad
        self.phase_e1_rad_nib = phase_e1_rad_nib
        self.phase_e2_rad_nib = phase_e2_rad_nib

        return self

    def unwrap(self):
        """Unwrap phase data (converted to radians) with scikit's unwrap_phase
        and prelude"""

        # With unwrap_phase (scikit)
        phase_e1_rad_scikit = unwrap_phase(self.phase_e1_rad)
        phase_e2_rad_scikit = unwrap_phase(self.phase_e2_rad)

        self.phase_e1_rad_scikit = phase_e1_rad_scikit
        self.phase_e2_rad_scikit = phase_e2_rad_scikit

        # With prelude
        phase_e1_rad_prelude_nib = prelude(abs=self.magnitude_e1_nib,
                                           phase=self.phase_e1_rad_nib,
                                           out=LOAD)['out']
        phase_e2_rad_prelude_nib = prelude(abs=self.magnitude_e2_nib,
                                           phase=self.phase_e2_rad_nib,
                                           out=LOAD)['out']

        self.phase_e1_rad_prelude = phase_e1_rad_prelude_nib.get_fdata()
        self.phase_e2_rad_prelude = phase_e2_rad_prelude_nib.get_fdata()

        return self

    def plot_raw(self, slice_to_show, suptitle):
        """Helper plotting function"""

        images = []
        images.append(self.magnitude_e1)
        images.append(self.magnitude_e2)
        images.append(self.phase_e1_rad)
        images.append(self.phase_e2_rad)
        images.append(self.phase_e1_rad_scikit)
        images.append(self.phase_e2_rad_scikit)
        images.append(self.phase_e1_rad_prelude)
        images.append(self.phase_e2_rad_prelude)

        titles = ['magnitude_e1',
                  'magnitude_e2',
                  'phase_e1_rad',
                  'phase_e2_rad',
                  'phase_e1_rad_scikit',
                  'phase_e2_rad_scikit',
                  'phase_e1_rad_prelude',
                  'phase_e2_rad_prelude']

        self.plot_common([2, 4], images, suptitle, titles, slice_to_show)

    def plot_common(self, mshape, images, suptitle, titles, slice_to_show):
        """General plotting function"""

        fig, axes = plt.subplots(mshape[0], mshape[1])
        for (ax, image, title) in zip(axes.flat, images, titles):

            this_slice = image[:, :, slice_to_show]
            roib_this_slice = self.roib[:, :, slice_to_show]

            # Calculate statistics for desired slice and generate a summary
            # string to show in the plots
            stats = ArrayStats(image, self.roib).calculate()
            summary_string = stats_summary(stats, slice_to_show)

            # Display image
            if SLICE_SPECIFIC_INTENSITY_RANGES:
                # not tested, ensure SLICE_SPECIFIC_INTENSITY_RANGES is FALSE
                im = ax.imshow(this_slice, cmap='gray',
                               vmin=stats['min']['2D'][slice_to_show][0],
                               vmax=stats['max']['2D'][slice_to_show][0])
            else:
                im = ax.imshow(this_slice, cmap='gray')

            ax.contour(roib_this_slice, colors='cyan', linewidths=0.2, alpha=0.5)
            # ax.format_coord = Formatter(im)
            ax.set_title(f"{title}\n{summary_string}", fontsize=10)
            ax.axis('off')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            plt.colorbar(im, cax=cax)
            # pixelinfo()
        fig.suptitle(f"{suptitle} - slice {slice_to_show}")

        # Make fullscreen
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()

        plt.show()


def stats_summary(statistics, slice_to_show):
    """Summarise statistics in a string"""

    min2 = statistics['min']['2D'][slice_to_show][0]
    max2 = statistics['max']['2D'][slice_to_show][0]
    n2 = statistics['n']['2D'][slice_to_show][0]
    mean2 = statistics['mean']['2D'][slice_to_show][0]
    median2 = statistics['median']['2D'][slice_to_show][0]

    range2 = max2-min2
    summary_string = (f'min={min2:.2f}, max={max2:.2f}, mean={mean2:.2f}\n'
                      f'median={median2:.2f}, range={range2:.2f}, n={n2:.0f}')
    return summary_string


def main():
    for dataset, filenames, slice_to_show, suptitle \
         in zip(DATASETS, FILENAMES_ALL, SLICES_TO_SHOW, SUPTITLES):

        dir_dataset = os.path.join(DIR_DATA, dataset)
        unwrapping_results = CompareUnwrapping(dir_dataset, filenames)
        unwrapping_results.plot_raw(slice_to_show, suptitle)


main()
