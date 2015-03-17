import unittest
import numpy as np
from cstructures.array import Array


class TestMap(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def test_simple_add(self):
        a = Array.rand(256, 256).astype(np.float32)
        b = Array.rand(256, 256).astype(np.float32)

        actual = a + b
        expected = np.add(a, b)
        self._check(actual, expected)

    def test_simple_add_scalar(self):
        a = Array.rand(256, 256).astype(np.float32)

        actual = a + 3.0
        expected = np.add(a, 3.0)
        self._check(actual, expected)

        actual = 3.0 + a
        self._check(actual, expected)

    def test_simple_mul(self):
        a = Array.rand(256, 256).astype(np.float32)
        b = Array.rand(256, 256).astype(np.float32)

        actual = a * b
        expected = np.multiply(a, b)
        self._check(actual, expected)

    def test_simple_mul_scalar(self):
        a = Array.rand(256, 256).astype(np.float32)

        actual = a * 3.0
        expected = np.multiply(a, 3.0)
        self._check(actual, expected)

        actual = 3.0 * a
        self._check(actual, expected)
