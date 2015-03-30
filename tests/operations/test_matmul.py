import unittest
from cstructures import Array
from cstructures.operations.matmul import matmul
import numpy as np


class TestMatMul(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def setUp(self):
        self.a = Array.rand(256, 256)
        self.b = Array.rand(256, 256)

    def test_square_dgemm(self):
        c = matmul(self.a, self.b, False, False)
        self._check(c, np.dot(self.a.T, self.b.T).T)
        self.fail()

    def test_transpose_a(self):
        c = matmul(self.a, self.b, True, False)
        self._check(c, np.dot(self.a, self.b.T).T)
