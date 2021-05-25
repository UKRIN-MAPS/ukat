import os
from ukat.data import fetch
import numpy as np
import pytest

DIR_DATA = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestFetch:

    def test_ge_dwi(self):

        # Check if folder exists
        assert os.path.exists(os.path.join(DIR_DATA, "dwi", "ge"))
        directory = os.path.join(DIR_DATA, "dwi", "ge")

        # Check if the following extensions exist
        assert any(f.endswith('.bval') for f in os.listdir(directory))
        assert any(f.endswith('.bvec') for f in os.listdir(directory))
        assert any(f.endswith('.nii.gz') for f in os.listdir(directory))

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

        # Check if folder exists
        assert os.path.exists(os.path.join(DIR_DATA, "dwi", "philips"))
        directory = os.path.join(DIR_DATA, "dwi", "philips")

        # Check if the following extensions exist
        assert any(f.endswith('.bval') for f in os.listdir(directory))
        assert any(f.endswith('.bvec') for f in os.listdir(directory))
        assert any(f.endswith('.nii.gz') for f in os.listdir(directory))

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

        # Check if folder exists
        assert os.path.exists(os.path.join(DIR_DATA, "dwi", "siemens"))
        directory = os.path.join(DIR_DATA, "dwi", "siemens")

        # Check if the following extensions exist
        assert any(f.endswith('.bval') for f in os.listdir(directory))
        assert any(f.endswith('.bvec') for f in os.listdir(directory))
        assert any(f.endswith('.nii.gz') for f in os.listdir(directory))

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

    def test_ge_r2star(self):

        # Check if folder exists
        assert os.path.exists(os.path.join(DIR_DATA, "r2star", "ge"))
        directory = os.path.join(DIR_DATA, "r2star", "ge")

        # Check if the following extensions exist
        assert any(f.endswith('.json') for f in os.listdir(directory))
        assert any(f.endswith('.nii.gz') for f in os.listdir(directory))

        # Test if the fetch function works
        magnitude, affine, echo_times = fetch.r2star_ge()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

    def test_philips_r2star(self):

        # Check if folder exists
        assert os.path.exists(os.path.join(DIR_DATA, "r2star", "philips"))
        directory = os.path.join(DIR_DATA, "r2star", "philips")

        # Check if the following extensions exist
        assert any(f.endswith('.json') for f in os.listdir(directory))
        assert any(f.endswith('.nii.gz') for f in os.listdir(directory))

        # Test if the fetch function works
        magnitude, affine, echo_times = fetch.r2star_philips()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

    def test_siemens_r2star(self):

        # Check if folder exists
        assert os.path.exists(os.path.join(DIR_DATA, "r2star", "siemens"))
        directory = os.path.join(DIR_DATA, "r2star", "siemens")

        # Check if the following extensions exist
        assert any(f.endswith('.json') for f in os.listdir(directory))
        assert any(f.endswith('.nii.gz') for f in os.listdir(directory))

        # Test if the fetch function works
        magnitude, affine, echo_times = fetch.r2star_siemens()

        # Check the format of the outputs
        assert isinstance(magnitude, np.ndarray)
        assert np.unique(np.isnan(magnitude)) != [True]
        assert isinstance(affine, np.ndarray)
        assert isinstance(echo_times, np.ndarray)
        assert len(np.shape(magnitude)) == 4
        assert np.shape(affine) == (4, 4)
        assert len(np.shape(echo_times)) == 1

    def test_ge_b0(self):

        # Check if folder exists
        assert os.path.exists(os.path.join(DIR_DATA, "b0", "ge"))
        directory = os.path.join(DIR_DATA, "b0", "ge")

        # Check if the following extensions exist
        assert any(f.endswith('.json') for f in os.listdir(directory))
        assert any(f.endswith('.nii.gz') for f in os.listdir(directory))

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

        # philips_1 has issues, so no test will be written here.
        # Should be included in the future in case a new dataset is added.

        # Check if folder exists
        assert os.path.exists(os.path.join(DIR_DATA, "b0", "philips_2"))
        directory = os.path.join(DIR_DATA, "b0", "philips_2")

        # Check if the following extensions exist
        assert any(f.endswith('.json') for f in os.listdir(directory))
        assert any(f.endswith('.nii.gz') for f in os.listdir(directory))

        # Test if the fetch function works
        magnitude, phase, affine, echo_times = fetch.b0_philips(2)

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
            magnitude, phase, affine, echo_times = fetch.b0_philips()

        # If an incorrect dataset_id is given
        with pytest.raises(ValueError):
            magnitude, phase, affine, echo_times = fetch.b0_philips(3)

    def test_siemens_b0(self):

        # Check if folder exists
        assert os.path.exists(os.path.join(DIR_DATA, "b0", "siemens_1"))
        directory = os.path.join(DIR_DATA, "b0", "siemens_1")

        # Check if the following extensions exist
        assert any(f.endswith('.json') for f in os.listdir(directory))
        assert any(f.endswith('.nii.gz') for f in os.listdir(directory))

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

        # Check if folder exists
        assert os.path.exists(os.path.join(DIR_DATA, "b0", "siemens_2"))
        directory = os.path.join(DIR_DATA, "b0", "siemens_2")

        # Check if the following extensions exist
        assert any(f.endswith('.json') for f in os.listdir(directory))
        assert any(f.endswith('.nii.gz') for f in os.listdir(directory))

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

    def test_philips_t1(self):

        # Check if folder exists
        assert os.path.exists(os.path.join(DIR_DATA, "t1", "philips_1"))
        directory = os.path.join(DIR_DATA, "t1", "philips_1")

        # Check if the following extensions exist
        assert any(f.endswith('.json') for f in os.listdir(directory))
        assert any(f.endswith('.nii.gz') for f in os.listdir(directory))

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

        # Check if folder exists
        assert os.path.exists(os.path.join(DIR_DATA, "t1", "philips_2"))
        directory = os.path.join(DIR_DATA, "t1", "philips_2")

        # Check if the following extensions exist
        assert any(f.endswith('.json') for f in os.listdir(directory))
        assert any(f.endswith('.nii.gz') for f in os.listdir(directory))

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
            magnitude, phase, affine, inversion_times, _ = fetch.t1_philips()

        # If an incorrect dataset_id is given
        with pytest.raises(ValueError):
            magnitude, phase, affine, inversion_times, _ = fetch.t1_philips(3)

    def test_philips_t2(self):

        # Check if folder exists
        assert os.path.exists(os.path.join(DIR_DATA, "t2", "philips_1"))
        directory = os.path.join(DIR_DATA, "t2", "philips_1")

        # Check if the following extensions exist
        assert any(f.endswith('.json') for f in os.listdir(directory))
        assert any(f.endswith('.nii.gz') for f in os.listdir(directory))

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

        # If no dataset_id is given
        with pytest.raises(TypeError):
            magnitude, affine, echo_times = fetch.t2_philips()

        # If an incorrect dataset_id is given
        with pytest.raises(ValueError):
            magnitude, affine, echo_times = fetch.t2_philips(3)
