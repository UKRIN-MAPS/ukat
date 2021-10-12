import numpy as np
import pytest

from ukat.data import fetch


class TestFetch:

    def test_ge_b0(self):
        # Test if the fetch function works
        magnitude, phase, affine, echo_times = fetch.b0_ge()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(phase, np.ndarray)
        assert np.unique(np.isnan(phase)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert len(np.shape(phase)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

    def test_philips_b0(self):

        # Test if the fetch function works
        magnitude, phase, affine, echo_times = fetch.b0_philips()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(phase, np.ndarray)
        assert np.unique(np.isnan(phase)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert len(np.shape(phase)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

    def test_siemens_b0(self):
        # Test if the fetch function works
        magnitude, phase, affine, echo_times = fetch.b0_siemens(1)

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(phase, np.ndarray)
        assert np.unique(np.isnan(phase)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert len(np.shape(phase)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

        # Test if the fetch function works
        magnitude, phase, affine, echo_times = fetch.b0_siemens(2)

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(phase, np.ndarray)
        assert np.unique(np.isnan(phase)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert len(np.shape(phase)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

        # If no dataset_id is given
        with pytest.raises(TypeError):
            magnitude, phase, affine, echo_times = fetch.b0_siemens()

        # If an incorrect dataset_id is given
        with pytest.raises(ValueError):
            magnitude, phase, affine, echo_times = fetch.b0_siemens(3)

    def test_ge_dwi(self):
        # Test if the fetch function works
        magnitude, affine, bvals, bvecs = fetch.dwi_ge()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(bvals, np.ndarray)
        assert isinstance(bvecs, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(bvals)) == 1
        assert len(np.shape(bvecs)) == 2
        assert (np.shape(bvecs)[0] == 3 or np.shape(bvecs)[1] == 3)

    def test_philips_dwi(self):

        # Test if the fetch function works
        magnitude, affine, bvals, bvecs = fetch.dwi_philips()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(bvals, np.ndarray)
        assert isinstance(bvecs, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(bvals)) == 1
        assert len(np.shape(bvecs)) == 2
        assert (np.shape(bvecs)[0] == 3 or np.shape(bvecs)[1] == 3)

    def test_siemens_dwi(self):
        # Test if the fetch function works
        magnitude, affine, bvals, bvecs = fetch.dwi_siemens()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(bvals, np.ndarray)
        assert isinstance(bvecs, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(bvals)) == 1
        assert len(np.shape(bvecs)) == 2
        assert (np.shape(bvecs)[0] == 3 or np.shape(bvecs)[1] == 3)

    def test_philips_t1(self):
        # Test if the fetch function works
        magnitude, phase, affine, inversion_times, tss = fetch.t1_philips(1)

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(phase, np.ndarray)
        assert np.unique(np.isnan(phase)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(inversion_times, np.ndarray)
        assert isinstance(tss, int)
        assert len(np.shape(magnitude)) == 4
        assert len(np.shape(phase)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(inversion_times)) == 1

        # Test if the fetch function works
        magnitude, phase, affine, inversion_times, tss = fetch.t1_philips(2)

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(phase, np.ndarray)
        assert np.unique(np.isnan(phase)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(inversion_times, np.ndarray)
        assert isinstance(tss, float)
        assert len(np.shape(magnitude)) == 4
        assert len(np.shape(phase)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(inversion_times)) == 1

        # If no dataset_id is given
        with pytest.raises(TypeError):
            magnitude, phase, affine, inversion_times, _ = \
                fetch.t1_philips()

        # If an incorrect dataset_id is given
        with pytest.raises(ValueError):
            magnitude, phase, affine, inversion_times, _ = \
                fetch.t1_philips(3)

    def test_philips_t1w(self):
        # Test if the fetch function works
        image, affine = fetch.t1w_volume_philips()

        # Check the format of the outputs
        assert isinstance(image, np.ndarray)
        assert np.unique(np.isnan(image)) != [True]
        assert isinstance(affine, np.ndarray)
        assert len(np.shape(image)) == 3
        assert np.shape(affine) == (4, 4)

    def test_philips_t2(self):
        # Test if the fetch function works
        magnitude, affine, echo_times = fetch.t2_philips(1)

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

        # If an incorrect dataset_id is given
        with pytest.raises(ValueError):
            magnitude, affine, echo_times = fetch.t2_philips(3)

    def test_ge_t2star(self):
        # Test if the fetch function works
        magnitude, affine, echo_times = fetch.t2star_ge()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

    def test_philips_t2star(self):
        # Test if the fetch function works
        magnitude, affine, echo_times = fetch.t2star_philips()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

    def test_siemens_t2star(self):
        # Test if the fetch function works
        magnitude, affine, echo_times = fetch.t2star_siemens()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

    def test_philips_t2w(self):
        # Test if the fetch function works
        image, affine = fetch.t2w_volume_philips()

        # Check the format of the outputs
        assert isinstance(image, np.ndarray)
        assert np.unique(np.isnan(image)) != [True]
        assert isinstance(affine, np.ndarray)
        assert len(np.shape(image)) == 3
        assert np.shape(affine) == (4, 4)
