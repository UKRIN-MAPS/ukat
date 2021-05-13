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
        fe1_json = pjoin(folder, '00009__3D_B0_map_VOL_e1.json')
        fe1_raw = pjoin(folder, '00009__3D_B0_map_VOL_e1.nii.gz')
        fe2_json = pjoin(folder, '00009__3D_B0_map_VOL_e2.json')
        fe2_raw = pjoin(folder, '00009__3D_B0_map_VOL_e2.nii.gz')
        return fe1_json, fe1_raw, fe2_json, fe2_raw
    if name == 'dwi_ge':
        files, folder = fetch_dwi_ge()
        fraw = pjoin(folder, '00014__Cor_DWI_RT.nii.gz')
        fjson = pjoin(folder, '00014__Cor_DWI_RT.json')
        fbval = pjoin(folder, '00014__Cor_DWI_RT.bval')
        fbvec = pjoin(folder, '00014__Cor_DWI_RT.bvec')
        return fraw, fjson, fbval, fbvec

    if name == 'dwi_philips':
        files, folder = fetch_dwi_philips()
        fraw = pjoin(folder, '03901__DWI_5slices.nii.gz')
        fjson = pjoin(folder, '03901__DWI_5slices.json')
        fbval = pjoin(folder, '03901__DWI_5slices.bval')
        fbvec = pjoin(folder, '03901__DWI_5slices.bvec')
        return fraw, fjson, fbval, fbvec

    if name == 'dwi_siemens':
        files, folder = fetch_dwi_siemens()
        fraw = pjoin(folder, '00042__trig_dwi_13b_06dir.nii.gz')
        fjson = pjoin(folder, '00042__trig_dwi_13b_06dir.json')
        fbval = pjoin(folder, '00042__trig_dwi_13b_06dir.bval')
        fbvec = pjoin(folder, '00042__trig_dwi_13b_06dir.bvec')
        return fraw, fjson, fbval, fbvec


def b0_ge():
    fe1_json, fe1_raw, fe2_json, fe2_raw = get_fnames('b0_ge')

    # Load magnitude, real and imaginary data and corresponding echo times
    magnitude = []
    real = []
    imaginary = []
    echo_list = []
    for file in [fe1_raw, fe2_raw]:
        data = nib.load(file)
        magnitude.append(data.get_fdata()[..., 0])
        real.append(data.get_fdata()[..., 1])
        imaginary.append(data.get_fdata()[..., 2])

    for file in [fe1_json, fe2_json]:
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
def dwi_ge():
    fraw, fjson, fbval, fbvec = get_fnames('dwi_ge')
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    raw = nib.load(fraw)

    data = raw.get_fdata()
    affine = raw.affine
    return data, affine, bvals, bvecs


def dwi_philips():
    fraw, fjson, fbval, fbvec = get_fnames('dwi_philips')
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    raw = nib.load(fraw)

    data = raw.get_fdata()
    affine = raw.affine
    return data, affine, bvals, bvecs


def dwi_siemens():
    fraw, fjson, fbval, fbvec = get_fnames('dwi_siemens')
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    raw = nib.load(fraw)

    data = raw.get_fdata()
    affine = raw.affine
    return data, affine, bvals, bvecs
