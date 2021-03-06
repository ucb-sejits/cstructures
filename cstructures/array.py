import numpy as np

from ctree.frontend import get_ast
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from collections import namedtuple
from ctree.transformations import PyBasicConversions
from ctree.transforms import ConstantFold
import ctree.c.nodes as C
from ctree.templates.nodes import StringTemplate
from ctree.nodes import Project
import ctypes as ct
import ast
import inspect
from functools import wraps
import itertools
import sys

ArrayCfg = namedtuple('ArrayCfg', ['shape', 'dtype'])


class Backend(ast.NodeTransformer):
    def __init__(self, arg_cfg, symbol_table):
        self.symbol_table = symbol_table
        self.arg_cfg = arg_cfg
        self.cfg_dict = {}
        self.loop_shape_map = {}
        self.defns = []
        self.includes = None
        self.loop_var_map = {}

    def visit_CFile(self, node):
        self.defns = []
        self.includes = set()
        node = super(Backend, self).generic_visit(node)
        for defn in self.defns:
            node.body.insert(0, defn)
        for include in self.includes:
            node.body.insert(
                0, StringTemplate("#include <{}>".format(include)))
        return node

    def visit_FunctionDecl(self, node):
        for param, cfg in zip(node.params, self.arg_cfg):
            if type(cfg) == ArrayCfg:
                param.type = np.ctypeslib.ndpointer(cfg.dtype, len(cfg.shape),
                                                    cfg.shape)()
            elif type(cfg) in {np.float32, float}:
                param.type = ct.c_float()
            elif type(cfg) in {int}:
                param.type = ct.c_int()
            else:
                # TODO: Generalize type inference or add support for all types
                raise NotImplementedError()
            self.cfg_dict[param.name] = cfg
        node.defn = list(map(self.visit, node.defn))
        return node

    def gen_loop_nest(self, loopvars, cfg):
        body = []
        node = C.For(C.Assign(C.SymbolRef(loopvars[0], ct.c_int()),
                              C.Constant(0)),
                     C.Lt(C.SymbolRef(loopvars[0]), C.Constant(cfg.shape[0])),
                     C.PostInc(C.SymbolRef(loopvars[0])),
                     body)
        curr_node = node
        for loopvar, dim in zip(loopvars[1:], cfg.shape[1:]):
            curr_node = C.For(C.Assign(C.SymbolRef(loopvar, ct.c_int()),
                                       C.Constant(0)),
                              C.Lt(C.SymbolRef(loopvar), C.Constant(dim)),
                              C.PostInc(C.SymbolRef(loopvar)),
                              [])
            body.append(curr_node)
            body = curr_node.body
        self.loop_shape_map[loopvars] = cfg.shape
        return node, curr_node

    def is_loop_by_index(self, node):
        return (isinstance(node, ast.For) and
                isinstance(node.iter, ast.Call) and
                isinstance(node.iter.func, ast.Attribute) and
                node.iter.func.attr == 'indices')

    def process_loop_options(self, keywords, outer):
        for key in keywords:
            if key.arg == 'parallel':
                outer.pragma = 'omp parallel for'
            elif key.arg == 'cache_block':
                self.includes.add("math.h")
                if sys.version_info > (3, 0):
                    if not isinstance(key.value, ast.NameConstant) or \
                            key.value.value not in {True, False}:
                        raise NotImplementedError(
                            "Unsupported argument to cache_block " +
                            ast.dump(key))
                    if key.value.value:
                        outer = CacheBlockLoopNest().visit(outer)
                else:
                    if not isinstance(key.value, ast.Name) or \
                            key.value.id not in {'True', 'False'}:
                        raise NotImplementedError(
                            "Unsupported argument to cache_block " +
                            ast.dump(key))
                    if key.value.id == 'True':
                        outer = CacheBlockLoopNest().visit(outer)
            elif key.arg == 'unroll':
                pass
            else:
                raise NotImplementedError(
                    "Unsupported keyword argument to indices " + ast.dump(key))
        return outer

    loop_var = -1

    def gen_loop_var(self):
        self.loop_var += 1
        return "x{}".format(self.loop_var)

    def visit_For(self, node):
        if self.is_loop_by_index(node):
            cfg = self.cfg_dict[node.iter.func.value.id]
            if isinstance(node.target, ast.Tuple):
                loopvars = tuple(var.id for var in node.target.elts)
            else:
                loopvars = ()
                for _ in cfg.shape:
                    loopvars += (self.gen_loop_var(), )
                self.loop_var_map[node.target.id] = loopvars
            outer, inner = self.gen_loop_nest(loopvars, cfg)
            inner.body = list(map(self.visit, node.body))
            if node.iter.keywords:
                outer = self.process_loop_options(node.iter.keywords, outer)
            return outer

        node.body = list(map(self.visit, node.body))
        return node

    def gen_loop_index(self, loopvars, shape):
        curr = C.SymbolRef(loopvars[-1])
        for i in reversed(range(len(loopvars) - 1)):
            curr = C.Add(
                C.Mul(C.SymbolRef(loopvars[i]),
                      C.Constant(np.prod(shape[i + 1:]))),
                curr
            )
        return curr

    def visit_SymbolRef(self, node):
        if node.name in self.symbol_table:
            return C.Constant(self.symbol_table[node.name])
        return node

    def visit_BinaryOp(self, node):
        if isinstance(node.op, C.Op.ArrayRef):
            if isinstance(node.left, C.SymbolRef):
                target = node.left.name
                if target in self.cfg_dict:
                    target = self.cfg_dict[target]
                    # if type(target) in {int, float}:
                    #     return C.Constant(target)
                    if isinstance(node.right, ast.Tuple):
                        loopvars = node.right.elts
                        loopvars = tuple(var.name for var in loopvars)
                        node.right = self.gen_loop_index(
                            loopvars, target.shape)
                    elif isinstance(node.right, C.SymbolRef):
                        if node.right.name in self.loop_var_map:
                            loopvars = self.loop_var_map[node.right.name]
                            node.right = self.gen_loop_index(
                                loopvars, target.shape)
                    return node
            if isinstance(node.left, ast.Attribute):
                if node.left.value.name in self.cfg_dict:
                    attr = getattr(self.cfg_dict[node.left.value.name],
                                   node.left.attr)
                    return C.Constant(attr[node.right.value])
                else:
                    raise NotImplementedError()
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return node

    def visit_FunctionCall(self, node):
        if node.func.name in {'min', 'max'}:
            node.func.name = "f" + node.func.name
            # TODO: Add support for all math funcs
            self.includes.add("math.h")
            return super(Backend, self).generic_visit(node)
        # FIXME: This is specific for handling a map function
        # do we have to generalize?
        node.args = [self.visit(arg) for arg in node.args]
        func_tree = get_ast(self.symbol_table[node.func.name])
        func_tree = PyBasicConversions().visit(func_tree).body[0]
        func_tree = self.visit(func_tree)
        func_tree.name = C.SymbolRef(node.func.name)
        func_tree.set_static()
        func_tree.set_inline()
        self.defns.append(func_tree)
        # FIXME: Infer type
        for p in func_tree.params:
            p.type = ct.c_float()
        func_tree.return_type = ct.c_float()
        return node


class CacheBlockLoopNest(ast.NodeTransformer):
    """
    Transformer that cache block all loop nests it finds.
    TODO: Support for variable blocking factors.
    TODO: Tuning integration (handle variable sized nests?)
    """
    def __init__(self):
        super(CacheBlockLoopNest, self).__init__()
        self.block_factor = 32
        self.inside_nest = False
        self.nest = []

    def gen_nest(self):
        ret_node = self.nest[0]
        ret_node.pragma = 'omp for'
        curr_node = ret_node
        for node in self.nest[1:-1]:
            curr_node.body[0] = node
            curr_node = node
        return ret_node

    def block_loop(self, node):
        loopvar = node.init.left.name
        loopvar += loopvar
        self.nest.insert(
            0,
            C.For(
                C.Assign(C.SymbolRef(loopvar, node.init.left.type),
                         node.init.right),
                C.Lt(C.SymbolRef(loopvar), node.test.right),
                C.AddAssign(C.SymbolRef(loopvar),
                            C.Constant(self.block_factor)),
                [None]
            )
        )
        node.init.right = C.SymbolRef(loopvar)
        node.test.right = C.FunctionCall(
            C.SymbolRef("fmin"),
            [C.Add(C.SymbolRef(loopvar),
                   C.Constant(self.block_factor)),
             node.test.right])

    def visit(self, node):
        if not self.inside_nest and not isinstance(node, C.For):
            raise Exception("CacheBlockLoopNest.visit must be called with the \
                            outer For node of a loop next, got node type \
                            {} instead.".format(node))
        start = node.init.right
        end = node.test.right
        if not isinstance(start, C.Constant) or \
                not isinstance(end, C.Constant):
            # Cache blocking only works over constant ranges for now
            return node
        if end.value - start.value < self.block_factor:
            return node
        if self.inside_nest:
            self.nest.append(node)
            if isinstance(node.body[0], C.For):
                self.visit(node.body[0])
            self.block_loop(node)
        else:
            if isinstance(node.body[0], C.For):
                self.inside_nest = True
                self.nest.append(node)
                self.visit(node.body[0])
                self.block_loop(node)
                outer = self.gen_nest()
                if node.pragma == 'omp parallel for':
                    node.pragma = 'omp for'
                    outer.pragma = 'omp parallel'
                return outer
            else:
                return node


class ConcreteFn(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type, output):
        self._c_function = self._compile(entry_name, proj, entry_type)
        self.output = output

    def __call__(self, *args, **kwargs):
        a = []
        # TODO: This is a dirty hack, we need a way to specify the number of
        # arguments or trimming arguments
        arglen = len(self._c_function.argtypes)
        if self.output:
            arglen -= 1
        for i in range(arglen):
            a.append(args[i])
        if self.output:
            a.append(self.output(args))
        self._c_function(*a)
        if self.output:
            return a[-1]


class SpecializedFn(LazySpecializedFunction):
    """
    A general specialized function.
    Supports index based loops on Arrays like
        for y, x in a.indices():

    Inline constants available in the current scope
        (currently only supports two stack frames, this could be adjusted by
         adding a lazy lookup method, but is in danger of having to traverse a
         large call stack)

    """
    def __init__(self, tree, symbol_table, output):
        super(SpecializedFn, self).__init__(tree)
        self.symbol_table = symbol_table
        self.output = output

    def args_to_subconfig(self, args, kwargs):
        arg_cfg = ()
        for arg in args:
            if isinstance(arg, Array):
                arg_cfg += (ArrayCfg(arg.shape, arg.dtype), )
            elif type(arg) in {int, float, np.float32}:
                arg_cfg += (arg, )
            else:
                raise Exception("Unsupport arg type {}".format(type(arg)))
        for key in kwargs:
            if type(arg) in {int, float}:
                arg_cfg += (kwargs[key], )
            else:
                raise Exception("Unsupport kwarg type {}".format(type(arg)))
        if self.output:
            arg = self.output(args)
            arg_cfg += (ArrayCfg(arg.shape, arg.dtype), )
        return arg_cfg

    def transform(self, tree, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        tree = PyBasicConversions().visit(tree)
        tree = Backend(arg_cfg, self.symbol_table).visit(tree)
        tree = ConstantFold().visit(tree)
        tree.name = self.original_tree.body[0].name
        return tree

    def finalize(self, files, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        entry_type = (None, )
        for cfg in arg_cfg:
            if isinstance(cfg, ArrayCfg):
                entry_type += (np.ctypeslib.ndpointer(cfg.dtype,
                                                      len(cfg.shape),
                                                      cfg.shape), )
            elif type(cfg) in {np.float32, float}:
                entry_type += (ct.c_float, )
            elif isinstance(cfg, int):
                entry_type += (ct.c_int, )
            else:
                raise NotImplementedError()
        entry_type = ct.CFUNCTYPE(*entry_type)
        return ConcreteFn(files[0].name,
                          Project(files), entry_type, self.output)

    def eval_symbolically(self, *args, **kwargs):
        # TODO: This should return a symbolic output, not an actual
        return self.output(args)


def specialize(fn=None, output=None):
    """
    Specializes function fn using SpecalizedFn.
    """
    if fn is None:
        def wrapped(fn):
            return specialize(fn, output)
        return wrapped

    frame = inspect.stack()[1][0]
    symbol_table = frame.f_locals
    for i in range(3):
        frame = frame.f_back
        if frame is None:  # End of stack
            break
        symbol_table.update(frame.f_locals)
    # FIXME: symbol_table prints out a huge dict, why??
    # TODO: We grab the last two frames, what to do if there's more?

    spec_fn = SpecializedFn(get_ast(fn), symbol_table, output)

    @wraps(fn)
    def fn(*args, **kwargs):
        return spec_fn(*args, **kwargs)
    fn._specializer = spec_fn
    return fn


def specialized_dispatch(fn=None, num_args=None):
    """
    Wraps a dispatch function which returns a specializer based on the
    arguments passed to the function.  Also exposed the ability to trim
    arguments to the final call by passing the num_args keyword.

    For example uses see convolution or pooling operators
    """
    if num_args:
        def wrapped(fn):
            fn = specialized_dispatch(fn)
            fn.num_args = num_args
            return fn
        return wrapped

    @wraps(fn)
    def wrapped(*args, **kwargs):
        trimmed_args = args
        if wrapped.num_args:
            trimmed_args = args[:wrapped.num_args]
        return fn(*args, **kwargs)(*trimmed_args, **kwargs)
    wrapped.specialized_dispatch = True
    wrapped.fn = fn
    wrapped.num_args = None
    return wrapped


def gen_array_output(args):
    return Array.zeros_like(args[0])


@specialize(output=gen_array_output)
def array_array_add(a, b, output):
    """ Elementwise array addition """
    for pt in output.indices(parallel=True):
        output[pt] = a[pt] + b[pt]


@specialize(output=gen_array_output)
def array_scalar_add(a, b, output):
    """ Array scalar addition """
    for pt in output.indices(parallel=True):
        output[pt] = a[pt] + b


@specialize(output=gen_array_output)
def array_array_mul(a, b, output):
    for pt in output.indices(parallel=True):
        output[pt] = a[pt] * b[pt]


@specialize(output=gen_array_output)
def array_scalar_mul(a, b, output):
    for pt in output.indices(parallel=True):
        output[pt] = a[pt] * b


@specialize(output=gen_array_output)
def array_array_div(a, b, output):
    for pt in output.indices(parallel=True):
        output[pt] = a[pt] / b[pt]


@specialize(output=gen_array_output)
def array_scalar_div(a, b, output):
    for pt in output.indices(parallel=True):
        output[pt] = a[pt] / b


@specialize(output=gen_array_output)
def array_array_sub(a, b, output):
    for pt in output.indices(parallel=True):
        output[pt] = a[pt] - b[pt]


@specialize(output=gen_array_output)
def array_scalar_sub(a, b, output):
    for pt in output.indices(parallel=True):
        output[pt] = a[pt] - b


@specialize(output=gen_array_output)
def transpose(matrix, output):
    """ Elementwise array addition """
    for y, x in output.indices(parallel=True):
        output[y, x] = matrix[x, y]


def smap(func):
    """
    Wraps func with a specializer that will map over an array and call func on
    each element.
    TODO: Define a spec for types of functions supported by map.
    """
    @wraps(func)
    @specialize(output=gen_array_output)
    def fn(a, output):
        for pt in output.indices():
            output[pt] = func(a[pt])
    return fn


def smap2(func):
    """
    Wraps func with a specializer that will map over an array and call func on
    each element.
    TODO: Define a spec for types of functions supported by map.
    """
    @wraps(func)
    @specialize(output=gen_array_output)
    def fn(a, b, output):
        for pt in output.indices():
            output[pt] = func(a[pt], b[pt])
    return fn


# def smap(func):
#     """
#     Wrap func with a specializer that will map over an array and call func on
#     each element.
#     TODO: Define a spec for types of functions supported by map.
#     """
#     @wraps(func)
#     @specialize
#     def fn(*args):
#         for y, x in output.indices():
#             args[-1][y, x] = func(*[arg[y, x] for arg in args[:-1]])
#     return fn


class Array(np.ndarray):
    """
    A thin wrapper around numpy arrays that provide specialized implementations
    of various operations.
    """
    def indices(self, block_factor=None, unroll=False):
        """
        Return an iterator over the indices of the array

        Allows for specializer writes to declare tuning parameters on the
        generated loop.  These arguments are ignored for python
        """
        return itertools.product(range(d) for d in self.shape)

    @staticmethod
    def empty(*args, **kwargs):
        return np.empty(*args, **kwargs).view(Array)

    @staticmethod
    def zeros(*args, **kwargs):
        return np.zeros(*args, **kwargs).view(Array)

    @staticmethod
    def zeros_like(*args, **kwargs):
        return np.zeros_like(*args, **kwargs).view(Array)

    @staticmethod
    def rand(*args, **kwargs):
        return np.random.rand(*args, **kwargs).view(Array)

    @staticmethod
    def standard_normal(*args, **kwargs):
        return np.random.standard_normal(*args, **kwargs).view(Array)

    @staticmethod
    def empty_like(*args, **kwargs):
        return np.empty_like(*args, **kwargs).view(Array)

    @staticmethod
    def ones(*args, **kwargs):
        return np.ones(*args, **kwargs).view(Array)

    @staticmethod
    def array(*args, **kwargs):
        return np.array(*args, **kwargs).view(Array)

    @staticmethod
    def ones_like(*args, **kwargs):
        return np.ones_like(*args, **kwargs).view(Array)

    @staticmethod
    def eye(*args, **kwargs):
        return np.eye(*args, **kwargs).view(Array)

    @staticmethod
    def fromstring(*args, **kwargs):
        return np.fromstring(*args, **kwargs).view(Array)

    @specialized_dispatch
    def __add__(self, b):
        """
        Dispatches the proper specialized addition operation based on the types
        of the inputs.
        """
        if isinstance(b, Array):
            return array_array_add
        elif type(b) in {np.float32, float, int}:
            return array_scalar_add
        raise NotImplementedError()

    @specialized_dispatch
    def __radd__(self, b):
        """
        Dispatches the proper specialized addition operation based on the types
        of the inputs.
        """
        if isinstance(b, Array):
            return array_array_add
        elif type(b) in {np.float32, float, int}:
            return array_scalar_add
        raise NotImplementedError()

    @specialized_dispatch
    def __sub__(self, b):
        if isinstance(b, Array):
            return array_array_sub
        elif type(b) in {np.float32, float, int}:
            return array_scalar_sub
        raise NotImplementedError()

    @specialized_dispatch
    def __rsub__(self, b):
        if isinstance(b, Array):
            return array_array_sub
        elif type(b) in {np.float32, float, int}:
            return array_scalar_sub
        raise NotImplementedError()

    @specialized_dispatch
    def __mul__(self, b):
        if isinstance(b, Array):
            return array_array_mul
        elif type(b) in {np.float32, float, int}:
            return array_scalar_mul
        raise NotImplementedError(type(b))

    @specialized_dispatch
    def __rmul__(self, b):
        if isinstance(b, Array):
            return array_array_mul
        elif type(b) in {np.float32, float, int}:
            return array_scalar_mul
        raise NotImplementedError()

    @specialized_dispatch
    def __div__(self, b):
        if isinstance(b, Array):
            return array_array_div
        elif type(b) in {np.float32, float, int}:
            return array_scalar_div
        raise NotImplementedError()

    @specialized_dispatch
    def __rdiv__(self, b):
        if isinstance(b, Array):
            return array_array_div
        elif type(b) in {np.float32, float, int}:
            return array_scalar_div
        raise NotImplementedError()
