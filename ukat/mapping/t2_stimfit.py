import os
import warnings

import nibabel as nib
import numpy as np
from numba import jit
from pathos.pools import ProcessPool
from scipy import optimize
from tqdm import tqdm

from .resources.t2_stimfit import rf_pulses


class StimFitModel:
    def __init__(self, mode='non_selective', n_comp=1, ukrin_vendor=None):
        if mode != 'non_selective' and mode != 'selective':
            raise ValueError(f'mode must be either "non_selective" or '
                             f'"selective". You specified {mode}.')
        self.mode = mode
        if n_comp not in [1, 2, 3]:
            raise ValueError(f'n_comp must be either 1, 2 or 3. You specified '
                             f'{n_comp}.')
        self.n_comp = n_comp
        if ukrin_vendor not in ['ge', 'philips', 'siemens']:
            raise warnings.warn('ukrin_vendor was not specified. Using '
                                'default pulse sequence parameters.')
        self.opt = dict()
        self.opt['mode'] = self.mode
        self.opt['esp'] = 10e-3
        self.opt['etl'] = 20
        self.opt['T1'] = 3

        self.opt['RFe'] = dict()
        self.opt['RFr'] = dict()
        if self.mode == 'selective':
            self.opt['Dz'] = [-0.5, 0.5]
            self.opt['Nz'] = 51
            self.opt['Nrf'] = 64
            self.opt['RFe'] = {'RF': [],
                               'tau': 2e-3,
                               'G': 0.5,
                               'phase': 0,
                               'ref': 1,
                               'alpha': [],
                               'angle': 90}
            self.opt['RFr'] = {'RF': [],
                               'tau': 2e-3,
                               'G': 0.5,
                               'phase': 90,
                               'ref': 0,
                               'alpha': [],
                               'angle': 180,
                               'FA_array': np.ones(self.opt['etl'])}
        # Curve fitting parameters
        self.opt['lsq'] = {'Ncomp': n_comp,
                           'xtol': 5e-4,
                           'ftol': 1e-9}
        if self.opt['lsq']['Ncomp'] == 1:
            self.opt['lsq']['X0'] = [0.06, 0.1, 1]  # [T2(sec), amp, B1]
            self.opt['lsq']['XU'] = [3, 1e+3, 1.8]
            self.opt['lsq']['XL'] = [0.015, 0, 0.2]
        elif self.opt['lsq']['Ncomp'] == 2:
            self.opt['lsq']['X0'] = [0.02, 0.1, 0.131, 0.1, 1]
            self.opt['lsq']['XU'] = [0.25, 1e+3, 3, 1e+3, 1.8]
            self.opt['lsq']['XL'] = [0.015, 0, 0.25, 0, 0.2]
        elif self.opt['lsq']['Ncomp'] == 3:
            self.opt['lsq']['X0'] = [0.02, 0.1, 0.036, 0.1, 0.131, 0.1, 1]
            self.opt['lsq']['XU'] = [0.035, 1e+3, 0.13, 1e3, 3, 1e+3, 1.8]
            self.opt['lsq']['XL'] = [0.015, 0, 0.035, 0, 0.13, 0, 0.2]

        if ukrin_vendor is not None:
            self._set_ukrin_vendor(ukrin_vendor)
        if self.mode == 'selective':
            self.opt['RFe'] = self._set_rf(self.opt['RFe'])
            self.opt['RFr'] = self._set_rf(self.opt['RFr'])

    def _set_ukrin_vendor(self, vendor):
        self.vendor = vendor
        self.opt['T1'] = 1.5
        self.opt['esp'] = 0.0129
        self.opt['etl'] = 10
        self.opt['RFr']['FA_array'] = np.ones(self.opt['etl'])
        if self.vendor == 'ge':
            self.opt['RFe']['tau'] = 2000 / 1e6
            self.opt['RFe']['G'] = 0.751599
            self.opt['RFr']['tau'] = 3136 / 1e6
            self.opt['RFr']['G'] = 0.276839
            self.opt['RFe']['RF'] = rf_pulses.ge_90
            self.opt['RFr']['RF'] = rf_pulses.ge_180
            self.opt['Dz'] = [0, 0.45]
        elif self.vendor == 'philips':
            self.opt['RFe']['tau'] = 3820 / 1e6
            self.opt['RFe']['G'] = 0.392
            self.opt['RFr']['tau'] = 6010 / 1e6
            self.opt['RFr']['G'] = 0.327
            self.opt['RFe']['RF'] = rf_pulses.philips_90
            self.opt['RFr']['RF'] = rf_pulses.philips_180
            self.opt['Dz'] = [0, 0.45]
        elif self.vendor == 'siemens':
            self.opt['RFe']['tau'] = 3072 / 1e6
            self.opt['RFe']['G'] = 0.417
            self.opt['RFr']['tau'] = 3000 / 1e6
            self.opt['RFr']['G'] = 0.326
            self.opt['RFe']['RF'] = rf_pulses.ge_90
            self.opt['RFr']['RF'] = rf_pulses.ge_180
            self.opt['Dz'] = [0, 0.5]

    def _set_rf(self, rf):
        dz = self.opt['Dz']
        nz = self.opt['Nz']
        nrf = self.opt['Nrf']

        gamma = 2 * np.pi * 42.575e6 / 10000  # Gauss
        z = np.linspace(dz[0], dz[1], nz)
        scale = rf['angle'] / (gamma * rf['tau'] * abs(np.sum(rf['RF'])) / len(
            rf['RF']) * 180 / np.pi)
        rf['RF'] *= scale

        m = np.zeros([3, nz])
        m[2, :] = 1
        rf['RF'] = 1e-4 * rf['RF']  # approximation for
        # small tip angle

        phi = gamma * rf['G'] * z * rf['tau'] / nrf
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cp_rf = np.cos(rf['phase'] * np.pi / 180)
        sp_rf = np.sin(rf['phase'] * np.pi / 180)
        theta_rf = gamma * rf['RF'] * rf['tau'] / nrf
        ct_rf = np.cos(theta_rf)
        st_rf = np.sin(theta_rf)

        for i in range(nrf):
            for j in range(nz):
                rz = np.array([[cphi[j], sphi[j], 0],
                               [-sphi[j], cphi[j], 0],
                               [0, 0, 1]])
                m[:, j] = np.dot(rz, m[:, j])

            r = np.array([[1, 0, 0],
                          [0, ct_rf[i], st_rf[i]],
                          [0, -st_rf[i], ct_rf[i]]])
            if rf['phase'] != 0:
                rz = np.array([[cp_rf, sp_rf, 0],
                               [-sp_rf, cp_rf, 0],
                               [0, 0, 1]])
                rzm = np.array([[cp_rf, -sp_rf, 0],
                                [sp_rf, cp_rf, 0],
                                [0, 0, 1]])
                r = np.dot(rzm, np.dot(r, rz))
            m = np.dot(r, m)

        if rf['ref'] > 0:
            psi = -rf['ref'] / 2 * gamma * rf['G'] * z * rf['tau']
            for j in range(nz):
                rz = np.array([[np.cos(psi[j]), np.sin(psi[j]), 0],
                               [-np.sin(psi[j]), np.cos(psi[j]), 0],
                               [0, 0, 1]])
                m[:, j] = np.dot(rz, m[:, j])

        rf['RF'] = 1e4 * rf['RF']
        rf['alpha'] = 1e4 * np.arccos(m[2, :])
        return rf


class T2StimFit:
    def __init__(self, pixel_array, affine, model,
                 mask=None, multithread='auto'):
        self.pixel_array = pixel_array
        self.shape = pixel_array.shape[:-1]
        self.n_vox = np.prod(self.shape)
        self.affine = affine
        self.model = model

        assert multithread is True \
               or multithread is False \
               or multithread == 'auto', f'multithreaded must be True,' \
                                         f'False or auto. You entered ' \
                                         f'{multithread}'
        if multithread == 'auto':
            if self.n_vox > 20:
                multithread = True
            else:
                multithread = False
        self.multithread = multithread

        # Generate a mask if there isn't one specified
        if mask is None:
            self.mask = np.ones(self.shape, dtype=bool)
        else:
            self.mask = mask
            # Don't process any nan values
        self.mask[np.isnan(np.sum(pixel_array, axis=-1))] = False
        self._fit()

    def to_nifti(self, output_directory=os.getcwd(), base_file_name='Output',
                 maps='all'):
        """Exports some of the T2 class attributes to NIFTI.

        Parameters
        ----------
        output_directory : string, optional
            Path to the folder where the NIFTI files will be saved.
        base_file_name : string, optional
            Filename of the resulting NIFTI. This code appends the extension.
            Eg., base_file_name = 'Output' will result in 'Output.nii.gz'.
        maps : list or 'all', optional
            List of maps to save to NIFTI. This should either the string "all"
            or a list of maps from ["t2", "m0", "b1",
            "r2", "mask"].
        """
        os.makedirs(output_directory, exist_ok=True)
        base_path = os.path.join(output_directory, base_file_name)
        if maps == 'all' or maps == ['all']:
            maps = ['t2', 'm0', 'b1', 'r2', 'mask']
        if isinstance(maps, list):
            for result in maps:
                if result == 't2' or result == 't2_map':
                    t2_nifti = nib.Nifti1Image(self.t2_map, affine=self.affine)
                    nib.save(t2_nifti, base_path + '_t2_map.nii.gz')
                elif result == 'm0' or result == 'm0_map':
                    m0_nifti = nib.Nifti1Image(self.m0_map, affine=self.affine)
                    nib.save(m0_nifti, base_path + '_m0_map.nii.gz')
                elif result == 'b1':
                    m0_err_nifti = nib.Nifti1Image(self.b1_map,
                                                   affine=self.affine)
                    nib.save(m0_err_nifti, base_path + '_b1_map.nii.gz')
                # TODO - add R-squared outputs
                # elif result == 'r2' or result == 'r2_map':
                #     r2_nifti = nib.Nifti1Image(T2.r2_map(self),
                #                                affine=self.affine)
                #     nib.save(r2_nifti, base_path + '_r2_map.nii.gz')
                elif result == 'mask':
                    mask_nifti = nib.Nifti1Image(self.mask.astype(np.uint16),
                                                 affine=self.affine)
                    nib.save(mask_nifti, base_path + '_mask.nii.gz')
        else:
            raise ValueError('No NIFTI file saved. The variable "maps" '
                             'should be "all" or a list of maps from '
                             '"["t2", "m0", "b1", "r2", '
                             '"mask"]".')

    def _fit(self):
        mask = self.mask.flatten()
        signal = self.pixel_array.reshape(self.n_vox, self.model.opt['etl'])
        idx = np.argwhere(mask).squeeze()
        signal = signal[idx, :]

        if self.multithread:
            with ProcessPool() as executor:
                results = executor.map(self._fit_signal, signal)
        else:
            results = list(tqdm(map(self._fit_signal, signal),
                                total=np.sum(self.mask)))
        t2 = np.array([result[0] for result in results])
        m0 = np.array([result[1] for result in results])
        b1 = np.array([result[2] for result in results])

        t2_map = np.zeros(self.n_vox)
        m0_map = np.zeros(self.n_vox)
        b1_map = np.zeros(self.n_vox)
        t2_map[idx] = t2
        m0_map[idx] = m0
        b1_map[idx] = b1
        self.t2_map = t2_map.reshape(self.shape)
        self.m0_map = m0_map.reshape(self.shape)
        self.b1_map = b1_map.reshape(self.shape)

    def _fit_signal(self, signal):
        if len(signal) != self.model.opt['etl']:
            raise Exception('Inconsistent echo train length')

        if self.model.opt['lsq']['Ncomp'] == 2:
            x = optimize.least_squares(self._residual2,
                                       self.model.opt['lsq']['X0'],
                                       args=(signal, self.model.opt,
                                             self.model.mode),
                                       bounds=(self.model.opt['lsq']['XL'],
                                               self.model.opt['lsq']['XU']),
                                       xtol=self.model.opt['lsq']['xtol'],
                                       ftol=self.model.opt['lsq']['ftol']).x
            t2, amp, b1 = [x[0], x[2]], [x[1], x[3]], x[5]

        elif self.model.opt['lsq']['Ncomp'] == 3:
            # TODO check with Leo this should be residual3 rather than
            #  residual2 as in original code!
            x = optimize.least_squares(self._residual3,
                                       self.model.opt['lsq']['X0'],
                                       args=(signal, self.model.opt,
                                             self.model.mode),
                                       bounds=(self.model.opt['lsq']['XL'],
                                               self.model.opt['lsq']['XU']),
                                       xtol=self.model.opt['lsq']['xtol'],
                                       ftol=self.model.opt['lsq']['ftol']).x
            t2, amp, b1 = [x[0], x[2], x[4]], [x[1], x[3], x[5]], x[6]

        else:
            x = optimize.least_squares(self._residual1,
                                       self.model.opt['lsq']['X0'],
                                       args=(signal, self.model.opt,
                                             self.model.mode),
                                       bounds=(self.model.opt['lsq']['XL'],
                                               self.model.opt['lsq']['XU']),
                                       xtol=self.model.opt['lsq']['xtol'],
                                       ftol=self.model.opt['lsq']['ftol']).x
            t2, amp, b1 = x
        return t2, amp, b1

    @staticmethod
    def _residual1(p, y, opt, mode):
        return y - _epgsig(p[0], p[2], opt, mode) * p[1]

    @staticmethod
    def _residual2(p, y, opt, mode):
        return y - (_epgsig(p[0], p[4], opt, mode) * p[1] -
                    _epgsig(p[2], p[4], opt, mode) * p[3])

    @staticmethod
    def _residual3(p, y, opt, mode):
        return y - (_epgsig(p[0], p[6], opt, mode) * p[1] -
                    _epgsig(p[4], p[6], opt, mode) * p[5] -
                    _epgsig(p[2], p[6], opt, mode) * p[3])


def _epgsig(t2, b1, opt, mode):
    sig = np.zeros(opt['etl'])
    if mode == 'non_selective':
        fa = np.pi / 180 * opt['RFr']['angle'] * np.array([
            opt['RFr']['FA_array']])
        sig = _epg(t2, b1, opt['T1'],
                   opt['esp'], fa,
                   opt['RFe']['angle'] * np.pi / 180)
    elif mode == 'selective':
        fa = np.array([opt['RFr']['alpha']]).T * \
             opt['RFr']['FA_array']
        m = _epg(t2, b1, opt['T1'], opt['esp'],
                 fa, opt['RFe']['alpha'])
        sig = np.sum(m, 0) / opt['Nz']
    return sig.ravel()


@jit(nopython=True)
def _epg(x2, b1, x1, esp, ar, ae):  # TE = 6.425ms. TR = 1500ms.   90, 175,
    # 145, 110, 110, 110.
    echo_intensity = np.zeros(ar.shape, dtype=np.float64)
    omiga = np.zeros((ar.shape[0], 3, 1 + 2 * ar.shape[1]),
                     dtype=np.float64)
    ar = b1 * ar
    ae = b1 * ae
    x2 = np.exp(-0.5 * esp / x2)
    x1 = np.exp(-0.5 * esp / x1)

    for i in range(omiga.shape[2]):
        if i == 0:
            omiga[:, 0, i] = np.sin(ae)
            omiga[:, 1, i] = np.sin(ae)
            omiga[:, 2, i] = np.cos(ae)
            continue
        omiga[:, 0, 1:i + 1] = omiga[:, 0, 0:i]
        omiga[:, 1, 0:i] = omiga[:, 1, 1:i + 1]
        omiga[:, 0, 0] = np.conj(omiga[:, 1, 0])
        omiga[:, 0:2, :] = x2 * omiga[:, 0:2, :]
        omiga[:, 2, :] = x1 * omiga[:, 2, :]
        omiga[:, 2, 0] += 1 - x1
        if i % 2 == 1:
            for runs in range(ar.shape[0]):
                ari = ar[runs, i // 2]
                t = np.array(
                    [[np.cos(0.5 * ari) ** 2, np.sin(0.5 * ari) ** 2,
                      np.sin(ari)],
                     [np.sin(0.5 * ari) ** 2, np.cos(0.5 * ari) ** 2,
                      -np.sin(ari)],
                     [-0.5 * np.sin(ari), +0.5 * np.sin(ari),
                      np.cos(ari)]], dtype=np.float64)
                omiga[runs, :, :] = np.dot(t, np.ascontiguousarray(
                    omiga[runs, :, :]))
        if i % 2 == 0:
            echo_intensity[:, i // 2 - 1] = omiga[:, 0, 0]
    return echo_intensity
