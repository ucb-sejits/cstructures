import unittest
import numpy as np
import inspect
from cstructures.array import Array, transpose, specialize
from cstructures.blas_transformers import dgemmify


dot = Array.dot


class TestDotFinder(unittest.TestCase):

    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def test_simple(self):

        def matrix_mult_sample():
            A = Array.array([[1, 0], [0, 1]])
            B = Array.array([[1, 1], [1, 1]])
            C = dot(A, B)
            return C

        expected = matrix_mult_sample()
        actual = dgemmify(matrix_mult_sample)()

        self._check(actual, expected)

    def test_multiple_nested(self):

        def matrix_mult_complex(A, B, C):
            return dot(dot(A, B), C)

        A = Array.rand(3, 3)
        B = Array.rand(3, 3)
        C = Array.rand(3, 3)

        expected = matrix_mult_complex(A, B, C)
        actual = dgemmify(matrix_mult_complex)(A, B, C)

        self._check(actual, expected)
