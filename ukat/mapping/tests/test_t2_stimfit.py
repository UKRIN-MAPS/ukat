import os
import shutil
import numpy as np
import numpy.testing as npt
import pytest
from ukat.data import fetch
from ukat.mapping.t2_stimfit import StimFitModel, T2StimFit, _epgsig, _epg
from ukat.utils import arraystats


class TestStimFitModel:
    def test_invalid_model(self):
        with pytest.raises(ValueError):
            model = StimFitModel(mode='invalid_model')

    def test_invalid_comp(self):
        with pytest.raises(ValueError):
            model = StimFitModel(n_comp=0)

        with pytest.raises(ValueError):
            model = StimFitModel(n_comp=4)

        with pytest.raises(ValueError):
            model = StimFitModel(n_comp='one')

    def test_invalid_vendor(self):
        with pytest.warns(UserWarning):
            model = StimFitModel(ukrin_vendor='brucker')

    def test_mode_switching(self):
        model = StimFitModel(mode='selective')
        assert model.mode == 'selective'
        assert model.opt['RFe']['angle'] == 90
        assert model.opt['Dz'] == [-0.5, 0.5]

        model = StimFitModel(mode='non_selective')
        assert model.mode == 'non_selective'
        assert model.opt['RFe']['angle'] == 90
        with pytest.raises(KeyError):
            model.opt['Dz']

    def test_n_comp_switching(self):
        model = StimFitModel(n_comp=1)
        assert model.n_comp == 1
        assert model.opt['lsq']['Ncomp'] == 1
        assert model.opt['lsq']['X0'][2] == 1

        model = StimFitModel(n_comp=2)
        assert model.n_comp == 2
        assert model.opt['lsq']['Ncomp'] == 2
        assert model.opt['lsq']['X0'][2] == 0.331

        model = StimFitModel(n_comp=3)
        assert model.n_comp == 3
        assert model.opt['lsq']['Ncomp'] == 3
        assert model.opt['lsq']['X0'][2] == 0.036

    def test_vendor_switching(self):
        model = StimFitModel(mode='selective', ukrin_vendor='ge')
        assert model.vendor == 'ge'
        assert model.opt['RFe']['tau'] == 2000 / 1e6

        model = StimFitModel(mode='selective', ukrin_vendor='philips')
        assert model.vendor == 'philips'
        assert model.opt['RFe']['tau'] == 3820 / 1e6

        model = StimFitModel(mode='selective', ukrin_vendor='siemens')
        assert model.vendor == 'siemens'
        assert model.opt['RFe']['tau'] == 3072 / 1e6

    def test_set_rf(self):
        model = StimFitModel(mode='selective', ukrin_vendor='ge')
        npt.assert_almost_equal(model.opt['RFe']['RF'][-1], 2.58327955e-07)
        npt.assert_almost_equal(model.opt['RFe']['alpha'][-1], 0.04164681,
                                decimal=5)
        npt.assert_almost_equal(model.opt['RFr']['RF'][-1], -3.55622605e-05)
        npt.assert_almost_equal(model.opt['RFr']['alpha'][-1], 1.97107315)

        model = StimFitModel(mode='selective', ukrin_vendor='philips')
        npt.assert_almost_equal(model.opt['RFe']['RF'][-1], 3.59081850e-04)
        npt.assert_almost_equal(model.opt['RFe']['alpha'][-1], 0.05002553,
                                decimal=5)
        npt.assert_almost_equal(model.opt['RFr']['RF'][-1], 0.00473865)
        npt.assert_almost_equal(model.opt['RFr']['alpha'][-1],  0.46764775,
                                decimal=5)

        model = StimFitModel(mode='selective', ukrin_vendor='siemens')
        npt.assert_almost_equal(model.opt['RFe']['RF'][-1], 1.68182263e-07)
        npt.assert_almost_equal(model.opt['RFe']['alpha'][-1], 0.07221162,
                                decimal=5)
        npt.assert_almost_equal(model.opt['RFr']['RF'][-1], -3.71744163e-05)
        npt.assert_almost_equal(model.opt['RFr']['alpha'][-1], 1.31133498)


class TestT2StimFit:
    image_ge, affine_ge, te_ge = fetch.t2_ge(1)
    image_ge = image_ge[::8, ::8, 2:4, :]  # Downsample to speed up tests
    image_philips, affine_philips, te_philips = fetch.t2_philips(2)
    image_philips = image_philips[::8, ::8, 2:4, :]
    image_siemens, affine_siemens, te_siemens = fetch.t2_siemens(1)
    image_siemens = image_siemens[::8, ::8, 2:4, :]

    # selective
    def test_selectiveness(self):
        # Selective
        model = StimFitModel(mode='selective', ukrin_vendor='ge')
        mapper = T2StimFit(self.image_ge, self.affine_ge, model)
        stats = arraystats.ArrayStats(mapper.t2_map).calculate()
        npt.assert_allclose([stats["mean"]["3D"], stats["std"]["3D"],
                             stats["min"]["3D"], stats["max"]["3D"]],
                            [110.635654, 292.978898, 15.0, 3000.0],
                            rtol=1e-6, atol=1e-4)

        # Non-selective
        model = StimFitModel(mode='non_selective', ukrin_vendor='ge')
        mapper = T2StimFit(self.image_ge, self.affine_ge, model)
        stats = arraystats.ArrayStats(mapper.t2_map).calculate()
        npt.assert_allclose([stats["mean"]["3D"], stats["std"]["3D"],
                             stats["min"]["3D"], stats["max"]["3D"]],
                            [111.737636, 302.09994, 15.0, 3000.0],
                            rtol=1e-6, atol=1e-4)

    # n_comp
    def test_n_comp(self):
        # Two Components
        model = StimFitModel(mode='selective', ukrin_vendor='ge', n_comp=2)
        mapper = T2StimFit(self.image_ge, self.affine_ge, model)
        stats = arraystats.ArrayStats(mapper.t2_map).calculate()
        npt.assert_allclose([stats["mean"]["4D"], stats["std"]["4D"],
                             stats["min"]["4D"], stats["max"]["4D"]],
                            [210.331084, 272.009048, 15.0, 2999.999999],
                            rtol=1e-6, atol=1e-4)

        # Three Components
        model = StimFitModel(mode='selective', ukrin_vendor='ge', n_comp=3)
        mapper = T2StimFit(self.image_ge, self.affine_ge, model)
        stats = arraystats.ArrayStats(mapper.t2_map).calculate()
        npt.assert_allclose([stats["mean"]["4D"], stats["std"]["4D"],
                             stats["min"]["4D"], stats["max"]["4D"]],
                            [346.479962, 845.632811, 15.0, 3000.0],
                            rtol=1e-6, atol=1e-4)

    # vendor
    def test_vendor(self):
        # Philips
        model = StimFitModel(mode='selective', ukrin_vendor='philips')
        mapper = T2StimFit(self.image_philips, self.affine_philips, model)
        stats = arraystats.ArrayStats(mapper.t2_map).calculate()
        npt.assert_allclose([stats["mean"]["3D"], stats["std"]["3D"],
                             stats["min"]["3D"], stats["max"]["3D"]],
                            [128.006434,  348.856183, 15.0, 2999.999999],
                            rtol=1e-6, atol=1e-4)

        # Siemens
        model = StimFitModel(mode='selective', ukrin_vendor='siemens')
        mapper = T2StimFit(self.image_siemens, self.affine_siemens, model)
        stats = arraystats.ArrayStats(mapper.t2_map).calculate()
        npt.assert_allclose([stats["mean"]["3D"], stats["std"]["3D"],
                             stats["min"]["3D"], stats["max"]["3D"]],
                            [145.373325, 453.853709, 15.0, 3000.0],
                            rtol=1e-5, atol=1e-2)
    # mask

    # threading

    # normalisation

    # to_nifti

class TestEpg:
    t2 = 0.1
    b1 = 0.95

    def test_selective(self):
        model = StimFitModel(mode='selective', ukrin_vendor='ge')
        sig = _epgsig(self.t2, self.b1, model.opt, model.mode)
        npt.assert_allclose(sig, np.array([0.53193464, 0.48718256, 0.41393849,
                                           0.37639148, 0.32247532, 0.29132453,
                                           0.25050307, 0.22604609, 0.19430487,
                                           0.1755666]),
                            rtol=1e-5, atol=1e-5)

    def test_non_selective(self):
        model = StimFitModel(mode='non_selective', ukrin_vendor='ge')
        sig = _epgsig(self.t2, self.b1, model.opt, model.mode)
        npt.assert_allclose(sig, np.array([0.87087025, 0.7713902, 0.6727603,
                                           0.59694589, 0.51965957, 0.46200336,
                                           0.40135138, 0.35760991, 0.30993556,
                                           0.27684428]),
                            rtol=1e-5, atol=1e-5)
