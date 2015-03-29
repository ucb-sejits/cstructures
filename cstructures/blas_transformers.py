# Writing Node transformers for an implementation of BLAS

import inspect
import sys
import ast

from ast import NodeTransformer, NodeVisitor, fix_missing_locations, Module
from ast import FunctionDef, Num, Name, Load, Call
from ctree import get_ast
from ctree.frontend import dump

# not explicitly used, but necessary when getting the new function
from cstructures.array import Array
from scipy.linalg.blas import dgemm
from collections import deque


def dgemmify(func):
    """
        This method takes a kernel function and uses DotOpFinder to
        convert any references to Array.dot (which is numpy.dot) to
        calls to scipy.linalg.blas.dgemm.

        :param: func (Function): the function to do this conversion on
        :return: a Function that does the same thing that func does,
                 except with dgemm calls instead of dot calls.
    """
    tree = get_ast(func)
    mod_tree = DotOpFinder().visit(tree)

    # Transpose finding
    Transpositioner = TranspositionFinder()
    mod_tree = Transpositioner.visit(mod_tree)
    mod_tree = NodeDeletor(Transpositioner.marked_for_deletion).visit(mod_tree)

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

    def __init__(self):
        self.transposes = {}

    def visit_Call(self, node):
        """
            This method handles visiting a FunctionCall, checking to see
            if the FunctionCall is a dot operation, and changes that into
            the BLAS equivalent (dgemm).

            :param: self (DotOpFinder): this DotOpFinder
            :param: node (FunctionCall): the FunctionCall node passed in
            :return: a new FunctionCall node that is a dgemm call if necessary
        """
        node.args = [self.visit(a) for a in node.args]
        fn = self.visit(node.func)

        try:
            func_name = fn.id
        except AttributeError:
            func_name = fn.attr

        if func_name is 'dot':                   # if it's a matrix multiply
            node.func.id = 'dgemm'
            node.args.insert(0, Num(n=1.0))      # adding the alpha parameter
            node.args.append(Num(n=1.0))         # adding the beta parameter

            # adding in the c parameter, and the transposition parameters
            try:
                node.args.append(ast.NameConstant(value=None))
                node.args.append(ast.NameConstant(value=False))
                node.args.append(ast.NameConstant(value=False))
            except AttributeError:
                node.args.append(ast.Name(id='None', ctx=Load()))
                node.args.append(ast.Name(id='False', ctx=Load()))
                node.args.append(ast.Name(id='False', ctx=Load()))

            return fix_missing_locations(node)
        else:
            return node


class TranspositionFinder(NodeTransformer):

    """
        This NodeTransformer gets rid of calls to numpy.transpose.
        Here's we're assuming that everything is line by line here, 
        so no nested calls.
    """

    def __init__(self):
        self.vars_transposed = {}
        self.nodes_transposed = {}
        self.marked_for_deletion = set()

        # TODO: we have to keep track of scoping... let's assume for now we're
        #      in the same scope throughout (the scoping case can be tested with
        #      a for loop through a list of matrices)

    def visit_Call(self, node):
        node.args = [self.visit(a) for a in node.args]
        fn = self.visit(node.func)

        try:
            func_name = fn.id
        except AttributeError:
            func_name = fn.attr

        if func_name is 'dgemm':
            for ind, arg in enumerate(node.args):
                if hasattr(arg, 'id') and arg.id in self.vars_transposed:

                    # retrive key attributes
                    target, value = self.nodes_transposed[arg.id]
                    transpose_of_arg = self.vars_transposed[arg.id]

                    self.nodes_transposed.pop(arg.id)
                    if transpose_of_arg != arg.id:
                        self.nodes_transposed.pop(transpose_of_arg)

                    # modifying the dgemm parameter matrix
                    node.args[ind].id = transpose_of_arg

                    # set transposition parameter of the dgemm call to True
                    try:
                        # py3
                        node.args[ind + 4] = ast.NameConstant(value=True)
                    except AttributeError:
                        node.args[ind + 4] = Name(id='True', ctx=Load())  # py2

                    # marking the transpose node(s) for removal
                    self.marked_for_deletion.add(target)
                    self.marked_for_deletion.add(value)
        return node

    def visit_Assign(self, node):

        # TODO:
        #
        #   (1) Handle multiple assignment for transpose
        #       (a) M, N = A.transpose(), B.transpose()

        target_value_list = [(self.visit(target), self.visit(value))
                             for target, value in self.parse_pairs(node)]

        # if we're assigning anything here, anything from the past no longer
        # counts
        for target, value in target_value_list:
            if target.id in self.vars_transposed:
                val = self.vars_transposed[target.id]
                self.remove_transpose(target.id)

                # remove
            if target.id in self.nodes_transposed:
                self.nodes_transposed.pop(target.id)
                self.nodes_transposed.pop(val)

        for target, value in target_value_list:
            if isinstance(value, Call):
                fn = self.visit(value.func)
                try:
                    func_name = fn.id
                except AttributeError:
                    func_name = fn.attr

                # if we're doing a transpose
                if func_name in {'T', 'transpose'}:
                    # register the transposes
                    self.add_transposes(target.id, value.args[0].id)

                    # register the nodes responsible for transposition
                    self.nodes_transposed[value.args[0].id] = (target, value)
                    self.nodes_transposed[target.id] = (target, value)

        return node

    #
    # Data Structure maintainance methods
    #

    def add_transposes(self, A, B):
        self.vars_transposed[A] = B
        self.vars_transposed[B] = A

    def remove_transpose(self, A):
        try:
            B = self.vars_transposed[A]
            self.vars_transposed.pop(A)
            self.vars_transposed.pop(B)
        except KeyError:
            pass

    #
    # Helper Methods from Ctree
    #

    def targets_to_list(self, targets):
        """parses target into nested lists"""
        res = []
        for elt in targets:
            if not isinstance(elt, (ast.List, ast.Tuple)):
                res.append(elt)
            elif isinstance(elt, (ast.Tuple, ast.List)):
                res.append(self.targets_to_list(elt.elts))
        return res

    def value_to_list(self, value):
        """parses value into nested lists for multiple assign"""
        res = []
        if not isinstance(value, (ast.List, ast.Tuple)):
            return value
        for elt in value.elts:
            if not isinstance(value, (ast.List, ast.Tuple)):
                res.append(elt)
            else:
                res.append(self.value_to_list(elt))
        return ast.List(elts=res)

    def pair_lists(self, targets, values):
        res = []
        queue = deque((target, values) for target in targets)
        sentinel = object()
        while queue:
            target, value = queue.popleft()
            if isinstance(target, list):
                #  target hasn't been completely unrolled yet
                for sub_target, sub_value in izip_longest(
                        target, value.elts, fillvalue=sentinel):
                    if sub_target is sentinel or \
                            sub_value is sentinel:
                        raise ValueError(
                            'Incorrect number of values to unpack')
                    queue.append((sub_target, sub_value))
            else:
                res.append((target, value))
        return res

    def parse_pairs(self, node):
        targets = self.targets_to_list(node.targets)
        values = self.value_to_list(node.value)
        return self.pair_lists(targets, values)


class NodeDeletor(NodeTransformer):

    def __init__(self, nodes_to_delete):
        self.nodes_to_delete = nodes_to_delete

    def visit(self, node):
        if node in self.nodes_to_delete:
            return None
        else:
            return super(NodeTransformer, self).visit(node)

    def visit_Assign(self, node):

        for i, targ in enumerate(node.targets):
            new_targ = self.visit(targ)
            if new_targ != None:
                node.targets[i] = new_targ

        node.value = self.visit(node.value)
        if node.targets is None or len(node.targets) == 0 or node.value is None:
            return None
        else:
            return node
