import unittest
import numpy as np
from cstructures.array import Array, smap, smap2


class TestMap(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def test_simple(self):
        a = Array.rand(256, 256).astype(np.float32)

        @smap
        def fn(x):
            if x > 0:
                return x
            else:
                return 0

        actual = fn(a)
        expected = np.copy(a)
        expected[expected < 0] = 0
        self._check(actual, expected)

    def test_two_inputs(self):
        a = Array.rand(256, 256).astype(np.float32) * 255.0 - 128.0
        b = Array.rand(256, 256).astype(np.float32) * 255.0 - 128.0
        negative_slope = 0.0

        @smap2
        def fn(x, y):
            if x > 0:
                return y
            else:
                return negative_slope * y

        actual = fn(a, b)
        expected = b
        expected[a <= 0] *= negative_slope
        self._check(actual, expected)


if __name__ == '__main__':
    unittest.main()
