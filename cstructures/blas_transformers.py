# Writing Node transformers for an implementation of BLAS

import inspect
import sys

from ast import NodeTransformer, fix_missing_locations, Module
from ast import FunctionDef, Num
from ctree import get_ast

# not explicitly used, but necessary when getting the new function
from cstructures.array import Array
from scipy.linalg.blas import dgemm


def dgemmify(func):
    '''
        This method takes a kernel function and uses DotOpFinder to
        convert any references to Array.dot (which is numpy.dot) to
        calls to scipy.linalg.blas.dgemm.

        :param: func (Function): the function to do this conversion on
        :return: a Function that does the same thing that func does,
                 except with dgemm calls instead of dot calls.
    '''
    tree = get_ast(func)
    mod_tree = DotOpFinder().visit(tree)

    # place the modified tree into a clean FunctionDef
    if sys.version_info >= (3, 0):
        mod_tree = Module(
            [FunctionDef(func.__name__, mod_tree.body[0].args,
                         mod_tree.body[0].body, [], None)]
        )
    else:
        mod_tree = Module(
            [FunctionDef(func.__name__, mod_tree.body[0].args,
                         mod_tree.body[0].body, [])]
        )

    mod_tree = fix_missing_locations(mod_tree)

    # compile the function and add it to current local namespace
    new_func_code = compile(
        mod_tree, filename=inspect.getsourcefile(func), mode='exec')
    exec(new_func_code)
    return locals()[func.__name__]


class DotOpFinder(NodeTransformer):

    def visit_Call(self, node):
        '''
            This method handles visiting a FunctionCall, checking to see
            if the FunctionCall is a dot operation, and changes that into
            the BLAS equivalent (dgemm).

            :param: self (DotOpFinder): this DotOpFinder
            :param: node (FunctionCall): the FunctionCall node passed in
            :return: a new FunctionCall node that is a dgemm call if necessary
        '''
        node.args = [self.visit(a) for a in node.args]
        fn = self.visit(node.func)

        try:
            func_name = fn.id
        except AttributeError:
            func_name = fn.attr

        if func_name is 'dot':                   # if it's a matrix multiply
            node.func.id = 'dgemm'
            node.args.insert(0, Num(n=1.0))  # adding the alpha parameter
            return fix_missing_locations(node)
        else:
            return node
