# writing Node transformers for an implementation of BLAS

import ast
from ctree.visitors import NodeTransformer
from scipy.linalg.blas import dgemm


class DotOpFinder(NodeTransformer):

    def __init__(self):
        pass

    # FIXME: Not sure if it should be visit_FunctionCall or visit_Call
    def visit_FunctionCall(self, node):
        '''
            This method handles visiting a FunctionCall, checking to see
            if the FunctionCall is a dot operation, and changes that into
            the BLAS equivalent (dgemm).

            :param: self (DotOpFinder): this DotOpFinder
            :param: node (FunctionCall): the FunctionCall node passed in
            :return: a new FunctionCall node that is a dgemm call if necessary
        '''
        func_name = node.func.name
        args_list = node.args

        if func_name in {'dot'}:            # if it's a matrix multiply
            args_list = [1.0] + args_list   # adding in the alpha parameter
            new_node = ast.FunctionCall(dgemm, args_list)
            return new_node
        else:
            return node
