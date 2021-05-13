import os
import nibabel as nib

from dipy.data.fetcher import _make_fetcher
from dipy.io import read_bvals_bvecs
from os.path import join as pjoin

if 'UKAT_HOME' in os.environ:
    ukat_home = os.environ['UKAT_HOME']
else:
    ukat_home = pjoin(os.path.expanduser('~'), '.ukat')

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
