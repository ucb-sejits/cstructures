import unittest
import numpy as np
from cstructures.array import Array, transpose


class TestMap(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def test_simple(self):
        a = Array.rand(256, 256).astype(np.float32)

        actual = Array.zeros(a.shape, np.float32)

        transpose(a, actual)
        expected = np.transpose(a)

        self._check(actual, expected)