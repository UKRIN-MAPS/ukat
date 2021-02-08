import tempfile
import filecmp
from os import path
import pytest
from ukat.utils.siemens import write_dvs


class TestWriteDvs:
    # Initialise two different test schemes, one read from diffusion_scheme.txt
    # in the `test_data` directory and another initalised here as a string, str
    scheme_txt = path.abspath(path.join(path.dirname(__file__), "..", "tests",
                                        "test_data", "diffusion_scheme.txt"))
    scheme_str = (" 0.70710678          0.0   0.70710678      0\n"
                  " 0.70710678          0.0   0.70710678      5\n"
                  " 0.70710678          0.0   0.70710678    100\n"
                  " 0.70710678          0.0   0.70710678    500\n"
                  "-0.70710678   0.70710678          0.0      5\n"
                  "-0.70710678   0.70710678          0.0    100\n"
                  "-0.70710678   0.70710678          0.0    500")

    # Initialise the paths to the .dvs files in test_data whose contents are
    # the gold standard to compare against the contents of the .dvs files
    # generated in the tests below
    expected_from_txt = path.abspath(path.join(path.dirname(__file__), "..",
                                               "tests", "test_data",
                                               "expected_from_txt.dvs"))
    expected_from_str = path.abspath(path.join(path.dirname(__file__), "..",
                                               "tests", "test_data",
                                               "expected_from_str.dvs"))

    def test_from_txt(self):
        # Note that the calls to write_dvs are done within a context manager to
        # remove temporary files generated during the tests
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_filepath = path.join(tmpdir, "tmp")
            dvs_path, _ = write_dvs(self.scheme_txt, tmp_filepath,
                                    normalization='none',
                                    coordinate_system='xyz',
                                    comment='This is a comment')

            # Ensure output file (dvs_path) matches the expected in test_data
            assert filecmp.cmp(self.expected_from_txt, dvs_path)

    def test_from_str(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_filepath = path.join(tmpdir, "tmp")
            dvs_path, _ = write_dvs(self.scheme_str, tmp_filepath,
                                    normalization='none',
                                    coordinate_system='xyz',
                                    comment='This is a comment')

            # Ensure output file (dvs_path) matches the expected in test_data
            assert filecmp.cmp(self.expected_from_str, dvs_path)

    def test_bad_normalization(self):
        with pytest.raises(ValueError):
            write_dvs(self.scheme_str, "dummy_name", normalization='bad_norm')

    def test_bad_coordinate_system(self):
        with pytest.raises(ValueError):
            write_dvs(self.scheme_str, "dummy_name", coordinate_system='ijk')

    def test_bad_comment(self):
        with pytest.raises(ValueError):
            write_dvs(self.scheme_str, "dummy_name", comment=None)
