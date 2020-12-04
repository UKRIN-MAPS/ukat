from ukat.mapping.dwi import make_gradient_scheme


class TestMakeGradientScheme:

    def test_one_bzero_true_with_bzero(self):
        bvals = [0, 5, 10]
        bvecs = [[1, 0, 1],
                 [-1, 1, 0]]
        output = make_gradient_scheme(bvals, bvecs, one_bzero=True)
        expected = (" 0.70710678          0.0   0.70710678      0\n"
                    " 0.70710678          0.0   0.70710678      5\n"
                    " 0.70710678          0.0   0.70710678     10\n"
                    "-0.70710678   0.70710678          0.0      5\n"
                    "-0.70710678   0.70710678          0.0     10")
        assert output == expected

    def test_one_bzero_true_without_bzero(self):
        bvals = [5, 10]
        bvecs = [[1, 0, 1],
                 [-1, 1, 0]]
        output = make_gradient_scheme(bvals, bvecs, one_bzero=True)
        expected = (" 0.70710678          0.0   0.70710678      0\n"
                    " 0.70710678          0.0   0.70710678      5\n"
                    " 0.70710678          0.0   0.70710678     10\n"
                    "-0.70710678   0.70710678          0.0      5\n"
                    "-0.70710678   0.70710678          0.0     10")
        assert output == expected

    def test_one_bzero_true_with_bzero_dont_normalize(self):
        bvals = [0, 5, 10]
        bvecs = [[1, 0, 1],
                 [-1, 1, 0]]
        output = make_gradient_scheme(bvals, bvecs, normalize=False,
                                      one_bzero=True)
        expected = ("          1            0            1      0\n"
                    "          1            0            1      5\n"
                    "          1            0            1     10\n"
                    "         -1            1            0      5\n"
                    "         -1            1            0     10")
        assert output == expected

    def test_one_bzero_false_with_bzero(self):
        bvals = [0, 5, 10]
        bvecs = [[1, 0, 1],
                 [-1, 1, 0]]
        output = make_gradient_scheme(bvals, bvecs, one_bzero=False)
        expected = (" 0.70710678          0.0   0.70710678      0\n"
                    " 0.70710678          0.0   0.70710678      5\n"
                    " 0.70710678          0.0   0.70710678     10\n"
                    "-0.70710678   0.70710678          0.0      0\n"
                    "-0.70710678   0.70710678          0.0      5\n"
                    "-0.70710678   0.70710678          0.0     10")
        assert output == expected

    def test_one_bzero_false_without_bzero(self):
        bvals = [5, 10]
        bvecs = [[1, 0, 1],
                 [-1, 1, 0]]
        output = make_gradient_scheme(bvals, bvecs, one_bzero=False)
        expected = (" 0.70710678          0.0   0.70710678      5\n"
                    " 0.70710678          0.0   0.70710678     10\n"
                    "-0.70710678   0.70710678          0.0      5\n"
                    "-0.70710678   0.70710678          0.0     10")
        assert output == expected
