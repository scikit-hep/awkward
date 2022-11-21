# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import sys

import awkward as ak


def mixin_class(registry, name=None):
    """
    Args:
        registry (dict): The destination behavior mapping registry. Typically,
            this would be the global registry #ak.behavior, but one may wish
            to register methods in an alternative way.
        name (str): The name to assign to the behaviour class.

    This decorator can be used to register a behavior mixin class.

    Any inherited behaviors will automatically be made available to the decorated
    class.

    See the "Mixin decorators" section of #ak.behavior for further details.
    """

    def register(cls):
        cls_name = cls.__name__
        if name is None:
            behavior_name = cls_name
        else:
            behavior_name = name

        record = type(
            cls_name + "Record",
            (cls, ak.highlevel.Record),
            {"__module__": cls.__module__},
        )
        setattr(sys.modules[cls.__module__], cls_name + "Record", record)
        registry[behavior_name] = record
        array = type(
            cls_name + "Array",
            (cls, ak.highlevel.Array),
            {"__module__": cls.__module__},
        )
        setattr(sys.modules[cls.__module__], cls_name + "Array", array)
        registry["*", behavior_name] = array
        for basecls in cls.mro():
            for method in basecls.__dict__.values():
                if hasattr(method, "_awkward_mixin"):
                    ufunc, rhs, transpose = method._awkward_mixin
                    if rhs is None:
                        registry.setdefault((ufunc, behavior_name), method)
                        continue
                    for rhs_name in list(rhs) + [behavior_name]:
                        registry.setdefault((ufunc, behavior_name, rhs_name), method)
                        if transpose is not None and rhs_name != behavior_name:
                            registry.setdefault(
                                (ufunc, rhs_name, behavior_name), transpose
                            )
                    if basecls.__name__ in rhs:
                        rhs.add(behavior_name)
        return cls

    return register


def mixin_class_method(ufunc, rhs=None, *, transpose=True):
    """
    Args:
        ufunc (numpy.ufunc): A universal function (or NEP18 callable) that is
            hooked in Awkward Array, i.e. it can be the first argument of a behavior.
        rhs (Set[type] or None): Set of right-hand side argument types, optional
            if wrapping a unary function. The left-hand side is expected to
            always be `self` of the parent class.
        transpose (bool): If true, automatically create a transpose signature
            (only makes sense for binary ufuncs).

    This decorator can be used to register a mixin class method.

    Using this decorator ensures that derived classes that are declared with the
    #ak.mixin_class decorator will also have the behaviors that this class has.
    """

    def register(method):
        if not isinstance(rhs, (set, type(None))):
            raise ak._errors.wrap_error(
                ValueError("expected a set of right-hand-side argument types")
            )
        if transpose and rhs is not None:

            def transposed(left, right):
                return method(right, left)

            # make a copy of rhs, we will edit it later
            method._awkward_mixin = (ufunc, set(rhs), transposed)
        else:
            method._awkward_mixin = (ufunc, rhs, None)
        return method

    return register
