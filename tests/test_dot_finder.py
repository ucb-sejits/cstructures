import unittest
import numpy as np
from cstructures.array import Array, transpose, specialize
import inspect

from cstructures.blas_transformers import dgemmify


dot = Array.dot

class TestDotFinder(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def test_simple(self):

    	@dgemmify
    	def matrix_mult_special():
    		A = Array.array([[1, 0], [0, 1]])
    		B = Array.array([[1, 1], [1, 1]])
    		C = dot(A, B)


    		# A = Array.eye(5)
    		# B = Array.ones_like(A)
    		# C = Array.dot(A, B)
    		return C

    	def matrix_mult_unspecial():
    		A = Array.eye(2)
    		B = Array.ones_like(A)
    		C = Array.dot(A, B)

    		# A = Array.eye([[1, 0], [0, 1]])
    		# B = Array.ones([[1, 1],[1, 1]])
    		# C = Array.dot(A, B)
    		return C

    	expected = matrix_mult_unspecial()


    	print ("MATRIX MULT: ", matrix_mult_special)
    	actual = matrix_mult_special()

    	# print (actual.func_code)
    	print (inspect.getsource(matrix_mult_special))



        self._check(actual, expected)
