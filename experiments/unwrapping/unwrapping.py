# Part of standard library or ukat
import os
import nibabel as nib
import numpy as np
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


class CompareUnwrapping():
    """Experimental class to perform unwrapping with several methods and plot
    the results in a figure montage"""

    def __init__(self, filepaths, te_diff):
        """Constructor for CompareUnwrapping class"""

        [self.magnitude_e1_path, self.phase_e1_path,
         self.magnitude_e2_path, self.phase_e2_path,
         self.roir_path, self.roil_path, self.b0_scanner_path] = filepaths

        self.te_diff = te_diff/1000

        self.load_data()
        self.convert_to_rad()
        self.unwrap()
        self.calculate_phase_diff()
        self.calculate_b0()

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

        # Regions of interest (r=right kidney, l=left kidney)
        roi = RegionOfInterest([self.roir_path, self.roil_path])
        self.roir = roi.data[0]
        self.roil = roi.data[1]

        # Scanner calculated B0 map
        b0_scanner_path = self.b0_scanner_path

        if not os.path.basename(b0_scanner_path):
            # if there is no scanner-generated B0 map, make it empty
            # this ensures the axes where the B0 map would be shown are deleted
            b0_scanner = ''
        else:
            b0_scanner_nib = nib.load(self.b0_scanner_path)
            b0_scanner = b0_scanner_nib.get_fdata()

        self.b0_scanner = b0_scanner

        return self

    def convert_to_rad(self):
        """Convert phase images into radians and "save" them into new
        NIfTI1Image objects (necessary for prelude)"""

        # Convert to rad and save as nib object
        phase_e1_rad = convert_to_pi_range(self.phase_e1)
        phase_e2_rad = convert_to_pi_range(self.phase_e2)
        phase_e1_rad_nib = nib.Nifti1Image(phase_e1_rad,
                                           self.phase_e1_nib.affine)
        phase_e2_rad_nib = nib.Nifti1Image(phase_e2_rad,
                                           self.phase_e2_nib.affine)

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

    def calculate_phase_diff(self):
        """Subtract phase images"""

        self.phase_diff = self.phase_e2_rad - self.phase_e1_rad
        self.phase_diff_scikit = (self.phase_e2_rad_scikit -
                                  self.phase_e1_rad_scikit)
        self.phase_diff_prelude = (self.phase_e2_rad_prelude -
                                   self.phase_e1_rad_prelude)

        return self

    def calculate_b0(self):
        """Calculate B0 maps"""

        d = (2*np.pi*self.te_diff)
        self.b0 = self.phase_diff / d
        self.b0_scikit = self.phase_diff_scikit / d
        self.b0_prelude = self.phase_diff_prelude / d

        return self

    # def print_report(self, slice_to_show):

    #     median2_r = statistics_r['median']['2D'][slice_to_show][0]

    #         this_slice = image[:, :, slice_to_show]
    #         roir_this_slice = self.roir[:, :, slice_to_show]
    #         roil_this_slice = self.roil[:, :, slice_to_show]

    #         # Calculate statistics for desired slice and generate a summary
    #         # string to show in the plots
    #         stats_r = ArrayStats(image, self.roir).calculate()
    #         stats_l = ArrayStats(image, self.roil).calculate()

    def roi_metrics(self, soi):
        """soi = slice of interest"""

        rois = [self.roir, self.roil]

        images = []                               #     R  L
        images.append(self.phase_e1_rad_scikit)   # = x[0, 9]
        images.append(self.phase_e2_rad_scikit)   # = x[1, 10]
        images.append(self.phase_e1_rad_prelude)  # = x[2, 11]
        images.append(self.phase_e2_rad_prelude)  # = x[3, 12]
        images.append(self.phase_diff_scikit)     # = x[4, 13]
        images.append(self.phase_diff_prelude)    # = x[5, 14]
        images.append(self.b0_scikit)             # = x[6, 15]
        images.append(self.b0_prelude)            # = x[7, 16]
        images.append(self.b0_scanner)            # = x[8, 17]

        # Create "x" vector that contains the median value (within the ROIs)
        # of many of the images generated during b0 mapping (including
        # intermediate results) in addition to some "metrics" derived from
        # these that may be useful for QA assessments.
        # Specifically, each element of x is as follows:
        # x[0]  - R_PU_S_1          : Right, unwrapped phase, scikit, echo 1
        # x[1]  - R_PU_S_2          : Right, unwrapped phase, scikit, echo 2
        # x[2]  - R_PU_P_1          : Right, unwrapped phase, prelude, echo 1
        # x[3]  - R_PU_P_2          : Right, unwrapped phase, prelude, echo 2
        # x[4]  - R_PD_S            : Right, phase difference scikit
        # x[5]  - R_PD_P            : Right, phase difference prelude
        # x[6]  - R_B0_S            : Right, b0, scikit
        # x[7]  - R_B0_P            : Right, b0, prelude
        # x[8]  - R_B0_O            : Right, b0, scanner
        # x[9]  - L_PU_S_1          : Left, unwrapped phase, scikit, echo 1
        # x[10] - L_PU_S_2          : Left, unwrapped phase, scikit, echo 2
        # x[11] - L_PU_P_1          : Left, unwrapped phase, prelude, echo 1
        # x[12] - L_PU_P_2          : Left, unwrapped phase, prelude, echo 2
        # x[13] - L_PD_S            : Left, phase difference scikit
        # x[14] - L_PD_P            : Left, phase difference prelude
        # x[15] - L_B0_S            : Left, b0, scikit
        # x[16] - L_B0_P            : Left, b0, prelude
        # x[17] - L_B0_O            : Left, b0, scanner
        # x[18] - (R_PD_P - R_PD_S) : Right, difference of phase differences
        #                             after unwrapping with prelude and scikit
        # x[19] - (L_PD_P - L_PD_S) : Left, difference of phase differences
        #                             after unwrapping with prelude and scikit
        # x[20] - (R_B0_S - L_B0_S) : Right-left B0 difference, scikit
        # x[21] - (R_B0_P - L_B0_P) : Right-left B0 difference, prelude
        # x[22] - (R_B0_O - L_B0_O) : Right-left B0 difference, scanner (_O -> [O]nline)
        # x[23] - Right-left B0 % difference (prelude vs. scikit)
        # x[24] - Right-left B0 % difference (prelude vs. scanner)
        x = []
        for roi in rois:
            for image in images:
                if isinstance(image, str) and not image:
                    x.append(np.nan)
                    continue
                stats = ArrayStats(image, roi).calculate()
                x.append(stats['median']['2D'][soi][0])

        x.append(x[5]-x[4])   # = x[18]
        x.append(x[14]-x[13]) # = x[19]

        x.append(x[6]-x[15])  # = x[20]
        x.append(x[7]-x[16])  # = x[21]
        x.append(x[8]-x[17])  # = x[22]

        x.append(100*((x[21]-x[20])/x[21]))  # = x[23]
        x.append(100*((x[21]-x[22])/x[21]))  # = x[24]

        return [round(elem, 2) for elem in x]

    def plot_all(self, slice_to_show, suptitle, roi_ir):
        """Helper plotting function

        roi_ir = roi intensity range
            if True normalises image intensity to min, max of ROIs
        """

        images = []
        images.append(self.magnitude_e1)
        images.append(self.magnitude_e2)
        images.append(self.phase_e1_rad)
        images.append(self.phase_e2_rad)
        images.append(self.phase_e1_rad_scikit)
        images.append(self.phase_e2_rad_scikit)
        images.append(self.phase_e1_rad_prelude)
        images.append(self.phase_e2_rad_prelude)
        images.append(self.phase_diff)
        images.append(self.phase_diff_scikit)
        images.append(self.phase_diff_prelude)
        images.append('')
        images.append(self.b0)
        images.append(self.b0_scikit)
        images.append(self.b0_prelude)
        images.append(self.b0_scanner)

        titles = ['magnitude_e1',
                  'magnitude_e2',
                  'phase_e1_rad',
                  'phase_e2_rad',
                  'phase_e1_rad_scikit',
                  'phase_e2_rad_scikit',
                  'phase_e1_rad_prelude',
                  'phase_e2_rad_prelude',
                  'phase_diff',
                  'phase_diff_scikit',
                  'phase_diff_prelude',
                  '',
                  'b0',
                  'b0_scikit',
                  'b0_prelude',
                  'b0_scanner']

        self.plot_common(slice_to_show, suptitle, roi_ir, [4, 4], images,
                         titles)

    def plot_common(self, slice_to_show, suptitle, roi_ir, mshape, images,
                    titles):
        """General plotting function"""

        fig, axes = plt.subplots(mshape[0], mshape[1], constrained_layout=True)

        for (ax, image, title) in zip(axes.flat, images, titles):

            if isinstance(image, str) and not image:
                fig.delaxes(ax)
                continue

            this_slice = image[:, :, slice_to_show]
            roir_this_slice = self.roir[:, :, slice_to_show]
            roil_this_slice = self.roil[:, :, slice_to_show]

            # Calculate statistics for desired slice and generate a summary
            # string to show in the plots
            stats_r = ArrayStats(image, self.roir).calculate()
            stats_l = ArrayStats(image, self.roil).calculate()

            summary_string_2 = summarise_stats(stats_r, stats_l, slice_to_show)

            if roi_ir:
                # Show images with intensity ranges normalised to min, max
                # of ROI intensities
                cmin = min([stats_r['min']['2D'][slice_to_show][0],
                            stats_l['min']['2D'][slice_to_show][0]])
                cmax = max([stats_r['max']['2D'][slice_to_show][0],
                            stats_l['max']['2D'][slice_to_show][0]])
                im = ax.imshow(this_slice, cmap='gray', vmin=cmin, vmax=cmax)
                ir_string = "intensity range normalisation: kidney ROIs"
            else:
                # Don't specify intensity range to display
                im = ax.imshow(this_slice, cmap='gray')
                ir_string = "intensity range normalisation: entire image"

            ax.contour(roir_this_slice, colors='red', linewidths=0.2, alpha=0.5)
            ax.contour(roil_this_slice, colors='blue', linewidths=0.2, alpha=0.5)

            ax.set_title(f"{title}", fontsize=9)
            ax.text(1.3, 0.2, summary_string_2, fontsize=8, transform=ax.transAxes)
            ax.axis('off')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            plt.colorbar(im, cax=cax)

        fig.suptitle(f"{suptitle} - slice {slice_to_show} - {ir_string}")

        # # Make fullscreen
        # mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()

        fig.canvas.toolbar.pack_forget()

        plt.show()


def summarise_stats(statistics_r, statistics_l, slice_to_show):
    """Summarise statistics in a string"""

    # Statistics for each kidney at the desired slice
    min2_r = statistics_r['min']['2D'][slice_to_show][0]
    max2_r = statistics_r['max']['2D'][slice_to_show][0]
    n2_r = statistics_r['n']['2D'][slice_to_show][0]
    mean2_r = statistics_r['mean']['2D'][slice_to_show][0]
    median2_r = statistics_r['median']['2D'][slice_to_show][0]

    min2_l = statistics_l['min']['2D'][slice_to_show][0]
    max2_l = statistics_l['max']['2D'][slice_to_show][0]
    n2_l = statistics_l['n']['2D'][slice_to_show][0]
    mean2_l = statistics_l['mean']['2D'][slice_to_show][0]
    median2_l = statistics_l['median']['2D'][slice_to_show][0]

    range2_r = max2_r-min2_r
    range2_l = max2_l-min2_l

    # Right/left differences (d)
    d_min2 = min2_r - min2_l
    d_max2 = max2_r - max2_l
    d_n2 = n2_r - n2_l
    d_mean2 = mean2_r - mean2_l
    d_median2 = median2_r - median2_l
    d_range2 = range2_r - range2_l

    return (f'Right kidney:\n'
            f'min={min2_r:.2f}, max={max2_r:.2f}\n'
            f'mean={mean2_r:.2f}, median={median2_r:.2f}\n'
            f'range={range2_r:.2f}, n={n2_r:.0f}\n'
            f'\n'
            f'Left kidney:\n'
            f'min={min2_l:.2f}, max={max2_l:.2f}\n'
            f'mean={mean2_l:.2f}, median={median2_l:.2f}\n'
            f'range={range2_l:.2f}, n={n2_l:.0f}\n'
            f'\n'
            f'Difference (R-L):\n'
            f'min={d_min2:.2f}, max={d_max2:.2f}\n'
            f'mean={d_mean2:.2f}, median={d_median2:.2f}\n'
            f'range={d_range2:.2f}, n={d_n2:.0f}')