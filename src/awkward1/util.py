# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import awkward1.highlevel


def mixin_class(registry):
    """
    Args:
        registry (dict):
            The destination behavior mapping registry. Typically, this would
            be the global registry ``awkward1.behavior``, but one may wish
            to register methods in an alternative way.

    This decorator can be used to register a behavior mixin class.
    Any inheirited behaviors will automatically be made available to the decorated class.
    """

    def register(cls):
        name = cls.__name__
        registry[name] = type(name + "Record", (cls, awkward1.highlevel.Record), {})
        registry["*", name] = type(name + "Array", (cls, awkward1.highlevel.Array), {})
        for basecls in cls.__mro__:
            for method in basecls.__dict__.values():
                if hasattr(method, "_awkward_mixin"):
                    ufunc, rhs, transpose = method._awkward_mixin
                    if rhs is None:
                        registry.setdefault((ufunc, name), method)
                        continue
                    for rhs_name in list(rhs) + [name]:
                        registry.setdefault((ufunc, name, rhs_name), method)
                        if transpose is not None and rhs_name != name:
                            registry.setdefault((ufunc, rhs_name, name), transpose)
                    if basecls.__name__ in rhs:
                        rhs.add(name)
        return cls

    return register


def mixin_class_method(ufunc, rhs=None, transpose=True):
    """
    Args:
        ufunc (numpy.ufunc):
            A universal function (or NEP18 callable) that is hooked in awkward1,
            i.e. it can be the first argument of a behavior
        rhs (Set[type] or None):
            Set of right-hand side argument types (leave None if unary function)
            The left-hand side is expected to always be ``self`` of the parent class
            If the function is not unary or binary, call for help :)
        transpose (bool):
            If true, autmatically create a transpose signature (only makes sense for binary ufuncs)

    This decorator can be used to register a mixin class method.
    Using this decorator ensures that derived classes that are declared
    with the `mixin_class` decorator will also have the behaviors that this class has.
    """

    def register(method):
        if not isinstance(rhs, (set, type(None))):
            raise ValueError("Expected a set of right-hand-side argument types")
        if transpose and rhs is not None:

            def transposed(left, right):
                return method(right, left)

            # make a copy of rhs, we will edit it later
            method._awkward_mixin = (ufunc, set(rhs), transposed)
        else:
            method._awkward_mixin = (ufunc, rhs, None)
        return method

    return register
