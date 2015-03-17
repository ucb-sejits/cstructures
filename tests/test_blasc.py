import unittest
import numpy as np
from numpy import dot
from numpy import transpose as T
from cstructures.array import Array
from ctree.frontend import get_ast
import sys
import ast
import inspect
import ctypes
from ctypes import c_int, c_float, c_size_t

import os
from sys import platform as _platform

if _platform == "linux" or _platform == "linux2":
    ext = "so"
elif _platform == "darwin":
    ext = "dylib"

_blaslib = ctypes.cdll.LoadLibrary("libcblas.{}".format(ext))


def gemm(A, B, C, alpha, beta, m, n, k):
    cblas_row_major = c_int(101)
    no_trans = c_int(111)
    m = c_int(int(m))
    n = c_int(int(n))
    k = c_int(int(k))
    alpha = c_float(alpha)
    beta = c_float(beta)
    A_ptr = A.ctypes.data_as(ctypes.c_void_p)
    B_ptr = B.ctypes.data_as(ctypes.c_void_p)
    C_ptr = C.ctypes.data_as(ctypes.c_void_p)

    _blaslib.cblas_sgemm(cblas_row_major, no_trans, no_trans, m, n, k,
                         alpha, A_ptr, k, B_ptr, n, beta, C_ptr, n)

def get_callable(tree, name, env):
    tree.body[0].decorator_list = []
    exec(compile(tree, filename="", mode="exec"), env, env)
    return env[tree.body[0].name]


def blasc(fn):
    frame = inspect.stack()[1][0]
    symbol_table = frame.f_locals
    symbol_table.update(frame.f_globals)
    symbol_table.update(frame.f_back.f_locals)
    symbol_table.update(frame.f_back.f_globals)
    tree = get_ast(fn)
    def wrapped(*args, **kwargs):
        print(args)
        print([arg.id for arg in tree.body[0].args.args])
        for index, arg in enumerate(tree.body[0].args.args):
            symbol_table[arg.id] = args[index]
        # Process tree
        return get_callable(tree, tree.body[0].name, symbol_table)(*args, **kwargs)
    return wrapped


class TestBlasc(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected)

    def test_simple(self):
        A = Array.rand(256, 256).astype(np.float32)
        x = Array.rand(256, 256).astype(np.float32)
        b = Array.rand(256, 256).astype(np.float32)

        @blasc
        def fn(A, x, b):
            v1 = T(A)
            v2 = dot(v1, x)
            v3 = v2 - b
            return v3

        self._check(fn(A, x, b), np.transpose(A).dot(x) - b)

    def test_blase(self):
        A = Array.rand(256, 256).astype(np.float32)
        x = Array.rand(256, 256).astype(np.float32)
        actual = Array.rand(256, 256).astype(np.float32)
        gemm(A, x, actual, 1.0, 0.0, 256, 256, 256)

        self._check(actual, np.dot(A, x))
