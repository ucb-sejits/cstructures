# from array import Array
# import numpy as np


def flatten_objects(objs):
    """
    Flattens an array of objects that each have the cstruct_flatten() method implemented
    a into an array or arrays.

    :note: Requires all objects in objs to have the cstruct_flatten() method implemented.
    """
    return [flatten_object(obj) for obj in objs]


def flatten_object(obj):
    """
    Flattens an object that has the cstruct_flatten() method implemented into an array.

    :note: Requires the object to have the cstruct_flatten() method implemented.
    """
    invert_op = getattr(obj, "cstruct_flatten", None)

    if invert_op is not None:
        arr = obj.cstruct_flatten()
        return arr		# do we want this to be an Array object, or a np object?
    else:
        # TODO: find a way to do this anyways...
        print "Exception: method cstruct_flatten() not defined"
