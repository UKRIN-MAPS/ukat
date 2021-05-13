import os
import nibabel as nib

from dipy.data.fetcher import _make_fetcher
from dipy.io import read_bvals_bvecs
from os.path import join as pjoin


if 'UKAT_HOME' in os.environ:
    ukat_home = os.environ['UKAT_HOME']
else:
    ukat_home = pjoin(os.path.expanduser('~'), '.ukat')

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


def get_fnames(name):
    if name == 'dwi_philips':
        files, folder = fetch_dwi_philips()
        fraw = pjoin(folder, '03901__DWI_5slices.nii.gz')
        fjson = pjoin(folder, '03901__DWI_5slices.json')
        fbval = pjoin(folder, '03901__DWI_5slices.bval')
        fbvec = pjoin(folder, '03901__DWI_5slices.bvec')
        return fraw, fjson, fbval, fbvec

def dwi_philips():
    fraw, fjson, fbval, fbvec = get_fnames('dwi_philips')
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    raw = nib.load(fraw)

    data = raw.get_fdata()
    affine = raw.affine
    return data, affine, bvals, bvecs