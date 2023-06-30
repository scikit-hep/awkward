# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from collections import ChainMap
from collections.abc import Mapping

import awkward as ak
from awkward._nplikes import ufuncs
from awkward._typing import Any


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
    list_name = layout.parameter("__list__")
    if isinstance(list_name, str):
        cls = behavior.get(list_name)
        if isinstance(cls, type) and issubclass(cls, Array):
            return cls
    deep_list_record_name = layout.purelist_parameters("__record__", "__list__")
    if isinstance(deep_list_record_name, str):
        cls = behavior.get(("*", deep_list_record_name))
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
    for cls in type(obj).__mro__:
        fcn = behavior.get(("__cast__", cls))
        if fcn is not None:
            return fcn
    return None


def find_ufunc_generic(ufunc, layout, behavior):
    behavior = overlay_behavior(behavior)
    custom = layout.parameter("__list__")
    if custom is None:
        custom = layout.parameter("__record__")
    if isinstance(custom, str):
        fcn = behavior.get((ufunc, custom))
        if fcn is None:
            fcn = behavior.get((ufuncs.ufunc, custom))
        return fcn
    else:
        return None


def find_ufunc(behavior, signature: tuple):
    if not any(s is None for s in signature):
        behavior = overlay_behavior(behavior)

        # Special case all strings or hashable types.
        if all(isinstance(x, str) for x in signature):
            return behavior.get(signature)
        else:
            for key, custom in behavior.items():
                if (
                    isinstance(key, tuple)
                    and len(key) == len(signature)
                    and key[0] == signature[0]
                    and all(
                        k == s
                        or (
                            isinstance(k, type)
                            and isinstance(s, type)
                            and issubclass(s, k)
                        )
                        for k, s in zip(key[1:], signature[1:])
                    )
                ):
                    return custom


def find_record_typestr(
    behavior: None | Mapping,
    parameters: None | Mapping[str, Any],
    default: str | None = None,
):
    if parameters is None:
        return default
    behavior = overlay_behavior(behavior)
    return behavior.get(("__typestr__", parameters.get("__record__")), default)


def find_array_typestr(
    behavior: None | Mapping,
    parameters: None | Mapping[str, Any],
    default: str | None = None,
):
    if parameters is None:
        return default
    behavior = overlay_behavior(behavior)
    return behavior.get(("__typestr__", parameters.get("__list__")), default)


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
