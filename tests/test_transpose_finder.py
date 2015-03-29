import unittest
import numpy as np
import inspect
from cstructures.array import Array, transpose, specialize
from cstructures.blas_transformers import dgemmify


dot = Array.dot


class TestTransposeFinder(unittest.TestCase):

    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def test_simple(self):

        def matrix_mult_sample():

            B = Array.array([[10, 3], [7, 4]])

            # transposition
            A = Array.array([[1, 2], [3, 4]])
            M = Array.transpose(A)
            C = dot(B, M)
            return C

        expected = matrix_mult_sample()
        actual = dgemmify(matrix_mult_sample)()

        self._check(actual, expected)

    def test_double_transpose(self):

        def matrix_mult_sample():

            A = Array.array([[1, 2], [3, 4]])
            B = Array.array([[10, 3], [7, 4]])

            # double transposition
            M = Array.transpose(A)
            N = Array.transpose(B)

            C = dot(N, M)
            return C

        expected = matrix_mult_sample()
        actual = dgemmify(matrix_mult_sample)()

        self._check(actual, expected)

    def test_transpose_into_buffer(self):

        def matrix_mult_sample():

            B = Array.array([[10, 3], [7, 4]])

            # transposition
            A = Array.array([[1, 2], [3, 4]])
            A = Array.transpose(A)
            C = dot(B, A)
            return C

        expected = matrix_mult_sample()
        actual = dgemmify(matrix_mult_sample)()

        self._check(actual, expected)

    def test_transpose_into_buffer_double(self):

        def matrix_mult_sample():

            A = Array.array([[1, 2], [3, 4]])
            B = Array.array([[10, 3], [7, 4]])

            # double transposition
            A = Array.transpose(A)
            B = Array.transpose(B)

            C = dot(A, B)
            return C

        expected = matrix_mult_sample()
        actual = dgemmify(matrix_mult_sample)()

        self._check(actual, expected)

    def test_mid_update(self):

        def matrix_mult_sample():

            A = Array.array([[1, 2], [3, 4]])
            B = Array.array([[10, 3], [7, 4]])

            M = Array.transpose(A)

            M = Array.array([[1, 1], [3, 0]])

            C = dot(M, B)
            return C

        expected = matrix_mult_sample()
        actual = dgemmify(matrix_mult_sample)()

        self._check(actual, expected)

    def test_mid_update_one_of_two(self):

        def matrix_mult_sample():

            A = Array.array([[1, 2], [3, 4]])
            B = Array.array([[10, 3], [7, 4]])

            M = Array.transpose(A)
            N = Array.transpose(B)

            M = Array.array([[1, 1], [3, 0]])

            C = dot(M, N)
            return C

        expected = matrix_mult_sample()
        actual = dgemmify(matrix_mult_sample)()

        self._check(actual, expected)

    def test_mid_update_double(self):

        def matrix_mult_sample():

            A = Array.array([[1, 2], [3, 4]])
            B = Array.array([[10, 3], [7, 4]])

            M = Array.transpose(A)
            N = Array.transpose(B)

            M = Array.array([[1, 1], [3, 0]])
            N = Array.array([[1, 0], [7, 0]])

            C = dot(M, N)
            return C

        expected = matrix_mult_sample()
        actual = dgemmify(matrix_mult_sample)()

        self._check(actual, expected)

    def test_after_update_double(self):

        def matrix_mult_sample():

            A = Array.array([[1, 2], [3, 4]])
            B = Array.array([[10, 3], [7, 4]])

            M = Array.transpose(A)
            N = Array.transpose(B)

            C = dot(M, N)

            # the double assignment after the dot call
            M = Array.array([[1, 1], [3, 0]])
            N = Array.array([[1, 0], [7, 0]])

            return C

        expected = matrix_mult_sample()
        actual = dgemmify(matrix_mult_sample)()

        self._check(actual, expected)

    def test_after_update_double_in_place(self):

        def matrix_mult_sample():

            A = Array.array([[1, 2], [3, 4]])
            B = Array.array([[10, 3], [7, 4]])

            A = Array.transpose(A)
            B = Array.transpose(B)

            C = dot(A, B)

            # the double transpose in place after the dot call
            A = Array.transpose(A)
            B = Array.transpose(B)

            return C

        expected = matrix_mult_sample()
        actual = dgemmify(matrix_mult_sample)()

        self._check(actual, expected)
