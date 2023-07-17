import os
import shutil
import numpy as np
import numpy.testing as npt
import pytest
from ukat.data import fetch
from ukat.mapping.t2_stimfit import StimFitModel, T2StimFit
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

