import numpy as np
import numpy.testing as npt
import os

from importlib import reload
from ukat.data import fetcher


def test_ukat_home():
    test_path = 'TEST_PATH'
    if 'UKAT_HOME' in os.environ:
        old_home = os.environ['UKAT_HOME']
        del os.environ['UKAT_HOME']
    else:
        old_home = None

    reload(fetcher)

    npt.assert_string_equal(fetcher.ukat_home,
                            os.path.join(os.path.expanduser('~'), '.ukat'))
    os.environ['UKAT_HOME'] = test_path
    reload(fetcher)
    npt.assert_string_equal(fetcher.ukat_home, test_path)

    # return to previous state
    if old_home:
        os.environ['UKAT_HOME'] = old_home


class TestFetch:

    def test_philips_dwi(self):

        # Test if the fetch function works
        magnitude, affine, bvals, bvecs = fetcher.dwi_philips()

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
