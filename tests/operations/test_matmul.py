import unittest
from cstructures import Array
from cstructures.operations.matmul import matmul
import numpy as np


class TestMatMul(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def test_square_dgemm(self):
        a = Array.rand(256, 256)
        b = Array.rand(256, 256)
        c = matmul(a, b)
        self._check(c, np.dot(a.T, b.T).T)
