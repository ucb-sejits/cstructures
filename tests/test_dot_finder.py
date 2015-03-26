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

    	def matrix_mult():
    		A = Array.array([[1, 0], [0, 1]])
    		B = Array.array([[1, 1], [1, 1]])
    		C = dot(A, B)
    		return C

    	expected = matrix_mult()
    	actual = dgemmify(matrix_mult)()

        self._check(actual, expected)
