# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from collections import ChainMap
from collections.abc import Mapping

import awkward as ak
from awkward._nplikes import ufuncs


def overlay_behavior(behavior: dict | None) -> Mapping:
    """
    Args:
        behavior: behavior dictionary, or None

    Return a ChainMap object that overlays the given behavior
    on top of the global #ak.behavior
    """
    if behavior is None:
        return ak.behavior
    return ChainMap(behavior, ak.behavior)


def get_array_class(layout, behavior):
    from awkward.highlevel import Array

    behavior = overlay_behavior(behavior)
    arr = layout.parameter("__array__")
    if isinstance(arr, str):
        cls = behavior.get(arr)
        if isinstance(cls, type) and issubclass(cls, Array):
            return cls
    deeprec = layout.purelist_parameter("__record__")
    if isinstance(deeprec, str):
        cls = behavior.get(("*", deeprec))
        if isinstance(cls, type) and issubclass(cls, Array):
            return cls
    return Array


def get_record_class(layout, behavior):
    from awkward.highlevel import Record

    behavior = overlay_behavior(behavior)
    rec = layout.parameter("__record__")
    if isinstance(rec, str):
        cls = behavior.get(rec)
        if isinstance(cls, type) and issubclass(cls, Record):
            return cls
    return Record


def find_record_reducer(reducer, layout, behavior):
    behavior = overlay_behavior(behavior)
    rec = layout.parameter("__record__")
    if isinstance(rec, str):
        return behavior.get((reducer.highlevel_function(), rec))


def find_custom_cast(obj, behavior):
    behavior = overlay_behavior(behavior)
    for key, fcn in behavior.items():
        if (
            isinstance(key, tuple)
            and len(key) == 2
            and key[0] == "__cast__"
            and isinstance(obj, key[1])
        ):
            return fcn
    return None


def find_custom_broadcast(layout, behavior):
    behavior = overlay_behavior(behavior)
    custom = layout.parameter("__array__")
    if not isinstance(custom, str):
        custom = layout.parameter("__record__")
    if not isinstance(custom, str):
        custom = layout.purelist_parameter("__record__")
    if isinstance(custom, str):
        for key, fcn in behavior.items():
            if (
                isinstance(key, tuple)
                and len(key) == 2
                and key[0] == "__broadcast__"
                and key[1] == custom
            ):
                return fcn
    return None


def find_ufunc_generic(ufunc, layout, behavior):
    behavior = overlay_behavior(behavior)
    custom = layout.parameter("__array__")
    if not isinstance(custom, str):
        custom = layout.parameter("__record__")
    if isinstance(custom, str):
        for key, fcn in behavior.items():
            if (
                isinstance(key, tuple)
                and len(key) == 2
                and (key[0] is ufunc or key[0] is ufuncs.ufunc)
                and key[1] == custom
            ):
                return fcn
    return None


def find_ufunc(behavior, signature):
    if not any(s is None for s in signature):
        behavior = overlay_behavior(behavior)
        for key, custom in behavior.items():
            if (
                isinstance(key, tuple)
                and len(key) == len(signature)
                and key[0] == signature[0]
                and all(
                    k == s
                    or (
                        isinstance(k, type) and isinstance(s, type) and issubclass(s, k)
                    )
                    for k, s in zip(key[1:], signature[1:])
                )
            ):
                return custom


def find_typestrs(behavior):
    behavior = overlay_behavior(behavior)
    out = {}
    for key, typestr in behavior.items():
        if (
            isinstance(key, tuple)
            and len(key) == 2
            and key[0] == "__typestr__"
            and isinstance(key[1], str)
            and isinstance(typestr, str)
        ):
            out[key[1]] = typestr
    return out


def find_typestr(parameters, typestrs):
    if parameters is not None:
        record = parameters.get("__record__")
        if record is not None:
            typestr = typestrs.get(record)
            if typestr is not None:
                return typestr
        array = parameters.get("__array__")
        if array is not None:
            typestr = typestrs.get(array)
            if typestr is not None:
                return typestr
    return None


def behavior_of(*arrays, **kwargs):
    from awkward.highlevel import Array, ArrayBuilder, Record

    behavior = kwargs.get("behavior")
    if behavior is not None:
        # An explicit 'behavior' always wins.
        return behavior

    copied = False
    highs = (
        Array,
        Record,
        ArrayBuilder,
    )
    for x in arrays[::-1]:
        if isinstance(x, highs) and x.behavior is not None:
            if behavior is None:
                behavior = x.behavior
            elif behavior is x.behavior:
                pass
            elif not copied:
                behavior = dict(behavior)
                behavior.update(x.behavior)
                copied = True
            else:
                behavior.update(x.behavior)
    return behavior
