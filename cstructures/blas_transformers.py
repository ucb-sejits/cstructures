# writing Node transformers for an implementation of BLAS

import inspect
from ast import NodeTransformer, Call, copy_location, fix_missing_locations

# from ctree.transformations import PyBasicConversions

from ctree import get_ast
from ctree.frontend import dump
from cstructures.array import Array, transpose, specialize
from ctree.c.nodes import FunctionCall, Constant
from ctree.visitors import NodeTransformer
from scipy.linalg.blas import dgemm
from numpy import dot
from dis import dis
import sys
import ast
from numpy import dot


def dgemmify(gina):
    tree = get_ast(gina)

    print dump(tree)
    # print dump(tree)
    # print "TREE: ", tree.body[0].body[2].value

    mod_tree = DotOpFinder().visit(tree)
    # print "MOD TREE: ", mod_tree.body[0].body[2].value
    mod_tree = fix_missing_locations(mod_tree)

    # print ("TREE:", mod_tree)
    # print ("TREE BODY:", mod_tree.body)
    # print ("TREE OTHER STUFF:", mod_tree.body[0].body[2].value.func.attr)


    if sys.version_info >= (3, 0):
        mod_tree = ast.Module(
            [ast.FunctionDef("ltn", mod_tree.body[0].args,
                             mod_tree.body[0].body, [], None)]
        )
    else:
        mod_tree = ast.Module(
            [ast.FunctionDef("ltn", mod_tree.body[0].args,
                             mod_tree.body[0].body, [])]
        )

    mod_tree = fix_missing_locations(mod_tree)


    # print("MOD TREE TYPE: ", type(mod_tree))


    # print("11111")
    print dump(mod_tree)
    # print("22222")

    loc = locals().copy()
    mod_tree = fix_missing_locations(mod_tree)

    new_func_code = compile(
        mod_tree, filename=inspect.getsourcefile(gina), mode='exec')

    # new_func_code = compile(
    #     mod_tree, filename=inspect.getsourcefile(gina), mode='exec')

    # exec(compile(mod_tree, filename="<string>", mode="exec"), symbol_table._env, symbol_table._env)
    # exec(compile(mod_tree, filename="string", mode="exec"))
    exec(new_func_code)  # , glob, loc)

    
    print set(locals().keys()) - set(loc.keys())
    print "RETURN: ", locals()['ltn'] 
    return locals()['ltn']

    #######################
    ## RANDOM CODE STUFF ##
    #######################

    ## LENNY's CODE ##
    # if sys.version_info >= (3, 0):
    #     tree = ast.Module(
    #         [ast.FunctionDef("ltn", mod_tree.body[0].params,
    #                          list(mod_tree.body), [], None)]
    #     )
    # else:
    #     tree = ast.Module(
    #         [ast.FunctionDef("ltn", mod_tree.body[0].params,
    #                          list(basic_block.body), [])]
    #     )

    # GETTING SYMBOL TABLES
    # loc = locals().copy()
    # glob = globals().copy()

    # loc = locals().copy()
    # glob = globals().copy()

    # dt = {'dgemmify': dgemmify}
    # print "helllo"
    # dis(new_func_code)

    # print "FUNC: ", func
    # print dt.keys()
    # print new_func_code.co_code
    # print "GLOBAL DIFFERENCE: ", set(globals().keys()) - set(loc.keys())
    # print "LOCAL DIFFERENCE: ", set(locals().keys()) - set(loc.keys())
    # print ("NEW TREE: ", new_tree.body[0].body[2].value)
    # new_func_code = compile(mod_tree, filename=inspect.getfile(gina), mode='exec')
    # print "adam"
    # print str(tree.body[0].body[2].value)

    # new_func = compile(mod_tree, filename=inspect.getfile(gina), mode='exec')
    # print("PRINT: ", new_func)
    # print("EVAL: ", exec(new_func))
    # env = {}#locals().copy()
    # env.update(globals())
    # env.update(locals())

    # print "HELLOOOOOO"
    # print(str(new_func))

    # env_copy = env.copy()
    # exec(new_func)
    # print env
    # print set(env.keys()) - set(env_copy.keys())

    # for key in set(env.keys()) - set(env_copy.keys()):
    #     pass
    # print("KEY: ", key, " VALUE: ", env[key])

    # print globals()
    # print ("FUNC NAME", env[])
    # return locals()['gina']
    # return lambda x, y: x + y #exec(new_func)


class DotOpFinder(NodeTransformer):

    # def __init__(self):
    #     super()

    # FIXME: Not sure if it should be visit_FunctionCall or visit_Call

    def visit_Call(self, node):
        '''
            This method handles visiting a FunctionCall, checking to see
            if the FunctionCall is a dot operation, and changes that into
            the BLAS equivalent (dgemm).

            :param: self (DotOpFinder): this DotOpFinder
            :param: node (FunctionCall): the FunctionCall node passed in
            :return: a new FunctionCall node that is a dgemm call if necessary
        '''
        print("VISITING AN ast.Call INSTANCE")
        # print "NODE NAME: ", node.func.attr

        args = [self.visit(a) for a in node.args]
        fn = self.visit(node.func)

        try:
            func_name = fn.id
        except AttributeError:
            func_name = fn.attr

        if func_name is 'dot':
            # if it's a matrix multiply
            print "DETECTED numpy.dot()"
            args_list = [1.0] + args   # adding in the alpha parameter
            # new_node = copy_location(node, Call(func=dgemm, args=args_list))
            
            node.func.id = 'dgemm'
            node.args.insert(0, ast.Num(n=1.0))
            # new_node = Call(func=node.func, args=args_list)
            # func_attr = node.func
            # func_attr.node
            # new_node = Call(func=dot, args=args)
            # new_node = copy_location(new_node, node)
            # new_
            node = fix_missing_locations(node)
            return node
            
            # node.func = dgemm
            # node.args = args_list


            return new_node

            # print "NEW NODE: ", new_node.func.attr
            return new_node
        else:
            return node
