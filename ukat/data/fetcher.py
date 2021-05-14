import glob
import json
import os
import nibabel as nib
import numpy as np

from dipy.data.fetcher import _make_fetcher
from dipy.io import read_bvals_bvecs
from os.path import join as pjoin

if 'UKAT_HOME' in os.environ:
    ukat_home = os.environ['UKAT_HOME']
else:
    ukat_home = pjoin(os.path.expanduser('~'), '.ukat')

fetch_b0_ge = _make_fetcher('fetch_b0_ge', pjoin(ukat_home, 'b0_ge'),
                            'https://zenodo.org/record/4758189/files/',
                            ['00009__3D_B0_map_VOL_e1.json',
                             '00009__3D_B0_map_VOL_e1.nii.gz',
                             '00009__3D_B0_map_VOL_e2.json',
                             '00009__3D_B0_map_VOL_e2.nii.gz'],
                            ['00009__3D_B0_map_VOL_e1.json',
                             '00009__3D_B0_map_VOL_e1.nii.gz',
                             '00009__3D_B0_map_VOL_e2.json',
                             '00009__3D_B0_map_VOL_e2.nii.gz'],
                            ['68496d356804e09ab36836a9f6a5c717',
                             'b8bd073521436c2abaef88c58c04d048',
                             '193bf6964aeb29b438ea7945b071900a',
                             '81efa61e7e0d47f897c054f80da9dfd1'],
                            doc='Downloading GE B0 data')

fetch_b0_philips = _make_fetcher('fetch_b0_philips', pjoin(ukat_home,
                                                           'b0_philips'),
                                 'https://zenodo.org/record/4758303/files/',
                                 ['01401__B0_map_expiration_volume_2DMS_'
                                  'product_e1.json',
                                  '01401__B0_map_expiration_volume_2DMS_'
                                  'product_e1.nii.gz',
                                  '01401__B0_map_expiration_volume_2DMS_'
                                  'product_e1_ph.json',
                                  '01401__B0_map_expiration_volume_2DMS_'
                                  'product_e1_ph.nii.gz',
                                  '01401__B0_map_expiration_volume_2DMS_'
                                  'product_e2.json',
                                  '01401__B0_map_expiration_volume_2DMS_'
                                  'product_e2.nii.gz',
                                  '01401__B0_map_expiration_volume_2DMS_'
                                  'product_e2_ph.json',
                                  '01401__B0_map_expiration_volume_2DMS_'
                                  'product_e2_ph.nii.gz'],
                                 ['01401__B0_map_expiration_volume_2DMS_'
                                  'product_e1.json',
                                  '01401__B0_map_expiration_volume_2DMS_'
                                  'product_e1.nii.gz',
                                  '01401__B0_map_expiration_volume_2DMS_'
                                  'product_e1_ph.json',
                                  '01401__B0_map_expiration_volume_2DMS_'
                                  'product_e1_ph.nii.gz',
                                  '01401__B0_map_expiration_volume_2DMS_'
                                  'product_e2.json',
                                  '01401__B0_map_expiration_volume_2DMS_'
                                  'product_e2.nii.gz',
                                  '01401__B0_map_expiration_volume_2DMS_'
                                  'product_e2_ph.json',
                                  '01401__B0_map_expiration_volume_2DMS_'
                                  'product_e2_ph.nii.gz'],
                                 ['d8bec9a5768144bf05b840e30d6b1892',
                                  '55e9337424fef25626feedc4875425c1',
                                  '7b206cbbc80f34ab84f49cf5076eb6f3',
                                  '5a4a24dd71164808c81798f6ab02ca3d',
                                  '988f7f8102242c76759c210c621d28d7',
                                  'e8deb3b9ee113375082103bdb669636d',
                                  '6dfe2def8d5d382389e00fa10c622075',
                                  '270890cf80ebd2d34a1f465e8e7d73f0'],
                                 doc='Downloading Philips B0 dataset')

fetch_b0_siemens_1 = _make_fetcher('fetch_b0_siemens_1', pjoin(ukat_home,
                                                               'b0_siemens_1'),
                                   'https://zenodo.org/record/4761921/files/',
                                   ['00010__bh_b0map_3D_default_e1.json',
                                    '00010__bh_b0map_3D_default_e1.nii.gz',
                                    '00010__bh_b0map_3D_default_e2.json',
                                    '00010__bh_b0map_3D_default_e2.nii.gz',
                                    '00011__bh_b0map_3D_default_e1.json',
                                    '00011__bh_b0map_3D_default_e1.nii.gz',
                                    '00011__bh_b0map_3D_default_e2.json',
                                    '00011__bh_b0map_3D_default_e2.nii.gz'],
                                   ['00010__bh_b0map_3D_default_e1.json',
                                    '00010__bh_b0map_3D_default_e1.nii.gz',
                                    '00010__bh_b0map_3D_default_e2.json',
                                    '00010__bh_b0map_3D_default_e2.nii.gz',
                                    '00011__bh_b0map_3D_default_e1.json',
                                    '00011__bh_b0map_3D_default_e1.nii.gz',
                                    '00011__bh_b0map_3D_default_e2.json',
                                    '00011__bh_b0map_3D_default_e2.nii.gz'],
                                   ['2050298aa605f9d3e4f5ee9c3bf528ac',
                                    'e1ef327345b6db34324c22bb04575b2e',
                                    '721e6a27ee5452be5f4e21879a2a4d96',
                                    '79a0ad622fbffcb880c78b2bcffaed5f',
                                    '68e72bdb4a4d572f46b7b454af600930',
                                    '2d058c228cbe1933c19ae2c4dd817e20',
                                    'a0c3ad5a196ce491450a0f03b0ff7b96',
                                    '0aec402fd96b9bbe35fce5f2e76d96b8'],
                                   doc='Downloading Siemens B0 dataset 1')

fetch_b0_siemens_2 = _make_fetcher('fetch_b0_siemens_2', pjoin(ukat_home,
                                                               'b0_siemens_2'),
                                   'https://zenodo.org/record/4761921/files/',
                                   ['00044__bh_b0map_fa3_default_e1.json',
                                    '00044__bh_b0map_fa3_default_e1.nii.gz',
                                    '00044__bh_b0map_fa3_default_e2.json',
                                    '00044__bh_b0map_fa3_default_e2.nii.gz',
                                    '00045__bh_b0map_fa3_default_e1.json',
                                    '00045__bh_b0map_fa3_default_e1.nii.gz',
                                    '00045__bh_b0map_fa3_default_e2.json',
                                    '00045__bh_b0map_fa3_default_e2.nii.gz'],
                                   ['00044__bh_b0map_fa3_default_e1.json',
                                    '00044__bh_b0map_fa3_default_e1.nii.gz',
                                    '00044__bh_b0map_fa3_default_e2.json',
                                    '00044__bh_b0map_fa3_default_e2.nii.gz',
                                    '00045__bh_b0map_fa3_default_e1.json',
                                    '00045__bh_b0map_fa3_default_e1.nii.gz',
                                    '00045__bh_b0map_fa3_default_e2.json',
                                    '00045__bh_b0map_fa3_default_e2.nii.gz'],
                                   ['9df7d245866251ed9793a44b85a9e37c',
                                    'a33342821059cd4556e4c839834bc184',
                                    '1224cb202da05ce07c4a7293e20b6de7',
                                    '4cd9c5e7b7f3a136a99cd1bd42594f31',
                                    'd252f9af49faff40dbd9e9744eea0441',
                                    'dbc7c97959d6d143ac7055d690dd5cc5',
                                    'acc0a9689cea167ab16d0f7303e69750',
                                    '83c7311a9321c5cca81913136a735237'],
                                   doc='Downloading Siemens B0 dataset 2')

fetch_dwi_ge = _make_fetcher('fetch_dwi_ge', pjoin(ukat_home, 'dwi_ge'),
                             'https://zenodo.org/record/4757819/files/',
                             ['00014__Cor_DWI_RT.nii.gz',
                              '00014__Cor_DWI_RT.json',
                              '00014__Cor_DWI_RT.bval',
                              '00014__Cor_DWI_RT.bvec'],
                             ['00014__Cor_DWI_RT.nii.gz',
                              '00014__Cor_DWI_RT.json',
                              '00014__Cor_DWI_RT.bval',
                              '00014__Cor_DWI_RT.bvec'],
                             ['c76cdc72e32ad41cb5c469a9ada5cb83',
                              'fb9943f4a905c28a098b15194ffe2e61',
                              '3890e970e58825983acdbfd8f07fa55d',
                              'a536341625a299743557e467772a3e46'],
                             doc='Downloading GE DWI data')

fetch_dwi_philips = _make_fetcher('fetch_dwi_philips', pjoin(ukat_home,
                                                             'dwi_philips'),
                                  'https://zenodo.org/record/4757139/files/',
                                  ['03901__DWI_5slices.nii.gz',
                                   '03901__DWI_5slices.json',
                                   '03901__DWI_5slices.bval',
                                   '03901__DWI_5slices.bvec'],
                                  ['03901__DWI_5slices.nii.gz',
                                   '03901__DWI_5slices.json',
                                   '03901__DWI_5slices.bval',
                                   '03901__DWI_5slices.bvec'],
                                  ['da96320b38c6b201cb858057b4b534b3',
                                   '229e04a00fb4336a47af603ff565dea9',
                                   '6816ed33bd087ef465bfde5a75c0c11b',
                                   'c70d2a49c003dd53d63b9d4cb9388cdb'],
                                  doc='Downloading Philips DWI data')

fetch_dwi_siemens = _make_fetcher('fetch_dwi_siemens', pjoin(ukat_home,
                                                             'dwi_siemens'),
                                  'https://zenodo.org/record/4757887/files/',
                                  ['00042__trig_dwi_13b_06dir.nii.gz',
                                   '00042__trig_dwi_13b_06dir.json',
                                   '00042__trig_dwi_13b_06dir.bval',
                                   '00042__trig_dwi_13b_06dir.bvec'],
                                  ['00042__trig_dwi_13b_06dir.nii.gz',
                                   '00042__trig_dwi_13b_06dir.json',
                                   '00042__trig_dwi_13b_06dir.bval',
                                   '00042__trig_dwi_13b_06dir.bvec'],
                                  ['1836b56ba028b5d5d41ae5f35313889a',
                                   'a043b4fb0721d3c38db462d433975d31',
                                   'a57ce54e88c154d06a34722eaabb60fb',
                                   '32d551e73ab6481972a6c8eab44f556d'],
                                  doc='Downloading Siemens DWI data')


def get_fnames(name):
    if name == 'b0_ge':
        files, folder = fetch_b0_ge()
        fnames = glob.glob(pjoin(folder, '*'))
        return fnames

    elif name == 'b0_philips':
        files, folder = fetch_b0_philips()
        fnames = glob.glob(pjoin(folder, '*'))
        return fnames

    elif name == 'b0_siemens_1':
        files, folder = fetch_b0_siemens_1()
        fnames = glob.glob(pjoin(folder, '*'))
        return fnames

    elif name == 'b0_siemens_2':
        files, folder = fetch_b0_siemens_2()
        fnames = glob.glob(pjoin(folder, '*'))
        return fnames

    elif name == 'dwi_ge':
        files, folder = fetch_dwi_ge()
        fnames = glob.glob(pjoin(folder, '*'))
        return fnames

    elif name == 'dwi_philips':
        files, folder = fetch_dwi_philips()
        fnames = glob.glob(pjoin(folder, '*'))
        return fnames

    elif name == 'dwi_siemens':
        files, folder = fetch_dwi_siemens()
        fnames = glob.glob(pjoin(folder, '*'))
        return fnames


def b0_ge():
    fnames = get_fnames('b0_ge')

    # Load magnitude, real and imaginary data and corresponding echo times
    magnitude = []
    real = []
    imaginary = []
    echo_list = []
    for file in fnames:
        if file.endswith(".nii.gz"):
            data = nib.load(file)
            magnitude.append(data.get_fdata()[..., 0])
            real.append(data.get_fdata()[..., 1])
            imaginary.append(data.get_fdata()[..., 2])

        elif file.endswith(".json"):
            # Retrieve list of echo times in the original order
            with open(file, 'r') as json_file:
                hdr = json.load(json_file)
            echo_list.append(hdr['EchoTime'])

    # Move echo dimension to 4th dimension
    magnitude = np.moveaxis(np.array(magnitude), 0, -1)
    real = np.moveaxis(np.array(real), 0, -1)
    imaginary = np.moveaxis(np.array(imaginary), 0, -1)

    # Calculate Phase image => tan-1(Im/Re)
    # np.negative is used to change the sign - as discussed with Andy Priest
    phase = np.negative(np.arctan2(imaginary, real))

    echo_list = np.array(echo_list)

    # Sort by increasing echo time
    sort_idxs = np.argsort(echo_list)
    echo_list = echo_list[sort_idxs]
    magnitude = magnitude[..., sort_idxs]
    phase = phase[..., sort_idxs]

    return magnitude, phase, data.affine, echo_list


def b0_philips():
    return _load_b0_siemens_philips(get_fnames('b0_philips'))


def b0_siemens(dataset_id):
    """Fetches b0/siemens_{dataset_id} dataset
        dataset_id : int
            Number of the dataset to load:
            - dataset_id = 1 to load "b0\siemens_1"
            - dataset_id = 2 to load "b0\siemens_2"
        Returns
        -------
        See outputs of _load_b0_siemens_philips
        """

    possible_dataset_ids = [1, 2]

    if dataset_id not in possible_dataset_ids:
        error_msg = f"`dataset_id` must be one of {possible_dataset_ids}"
        raise ValueError(error_msg)

    if dataset_id == 1:
        return _load_b0_siemens_philips(get_fnames('b0_siemens_1'))
    elif dataset_id == 2:
        return _load_b0_siemens_philips(get_fnames('b0_siemens_2'))


def dwi_ge():
    fnames = get_fnames('dwi_ge')
    bval_path = [f for f in fnames if f.endswith('.bval')][0]
    bvec_path = [f for f in fnames if f.endswith('.bvec')][0]
    nii_path = [f for f in fnames if f.endswith('.nii.gz')][0]
    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    raw = nib.load(nii_path)

    data = raw.get_fdata()
    affine = raw.affine
    return data, affine, bvals, bvecs


def dwi_philips():
    fnames = get_fnames('dwi_philips')

    bval_path = [f for f in fnames if f.endswith('.bval')][0]
    bvec_path = [f for f in fnames if f.endswith('.bvec')][0]
    nii_path = [f for f in fnames if f.endswith('.nii.gz')][0]
    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    raw = nib.load(nii_path)

    data = raw.get_fdata()
    affine = raw.affine
    return data, affine, bvals, bvecs


def dwi_siemens():
    fnames = get_fnames('dwi_siemens')

    bval_path = [f for f in fnames if f.endswith('.bval')][0]
    bvec_path = [f for f in fnames if f.endswith('.bvec')][0]
    nii_path = [f for f in fnames if f.endswith('.nii.gz')][0]
    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    raw = nib.load(nii_path)

    data = raw.get_fdata()
    affine = raw.affine
    return data, affine, bvals, bvecs


def _load_b0_siemens_philips(filepaths):
    """General function to retrieve siemens b0 data from list of filepaths
    Returns
    -------
    numpy.ndarray
        image data - Magnitude
    numpy.ndarray
        image data - Phase
    numpy.ndarray
        affine matrix for image data
    numpy.ndarray
        array of echo times, in seconds
    """
    # Load magnitude, real and imaginary data and corresponding echo times
    data = []
    affines = []
    image_types = []
    echo_times = []

    for filepath in filepaths:

        if filepath.endswith(".nii.gz"):
            # Load data in NIfTI files
            nii = nib.load(filepath)
            data.append(nii.get_fdata())
            affines.append(nii.affine)

            # Load necessary information from corresponding .json files
            json_path = filepath.replace(".nii.gz", ".json")
            with open(json_path, 'r') as json_file:
                hdr = json.load(json_file)
                image_types.append(hdr['ImageType'])
                echo_times.append(hdr['EchoTime'])

    # Sort by increasing echo time
    sort_idxs = np.argsort(echo_times)
    data = np.array([data[i] for i in sort_idxs])
    echo_times = np.array([echo_times[i] for i in sort_idxs])
    image_types = [image_types[i] for i in sort_idxs]

    # Move measurements (time) dimension to 4th dimension
    data = np.moveaxis(data, 0, -1)

    # Separate magnitude and phase images
    magnitude_idxs = ["M" in i for i in image_types]
    phase_idxs = ["P" in i for i in image_types]

    magnitude = data[..., magnitude_idxs]
    phase = data[..., phase_idxs]

    echo_times_magnitude = echo_times[magnitude_idxs]
    echo_times_phase = echo_times[phase_idxs]

    # Assign unique echo times for output
    echo_times_are_equal = (echo_times_magnitude == echo_times_phase).all()
    if echo_times_are_equal:
        echo_times = echo_times_magnitude
    else:
        raise ValueError("Magnitude and phase echo times must be equal")

    # If all affines are equal, initialise the affine for output
    affines_are_equal = (np.array([i == affines[0] for i in affines])).all()
    if affines_are_equal:
        affine = affines[0]
    else:
        raise ValueError("Affine matrices of input data are not all equal")

    return magnitude, phase, affine, echo_times