import unittest
import numpy as np
from numpy import dot
from numpy import transpose as T
from cstructures.array import Array
from ctree.frontend import get_ast
import sys
import ast
import inspect


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
    print(symbol_table)
    tree = get_ast(fn)
    def wrapped(*args, **kwargs):
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

