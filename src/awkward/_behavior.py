# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from collections import ChainMap
from collections.abc import Mapping

import awkward as ak
from awkward._nplikes import ufuncs
from awkward._typing import JSONMapping


def overlay_behavior(behavior: Mapping | None) -> Mapping:
    """
    Args:
        behavior: behavior dictionary, or None

    Return a ChainMap object that overlays the given behavior
    on top of the global #ak.behavior
    """
    if behavior is None:
        return ak.behavior
    return ChainMap(behavior, ak.behavior)


def get_array_name(parameters: JSONMapping | None) -> str | None:
    if parameters is None:
        return None
    for param in "__name__", "__array__":
        name = parameters.get(param)
        if name is not None:
            return name
    return None


def get_record_name(parameters: JSONMapping | None) -> str | None:
    if parameters is None:
        return None
    return parameters.get("__record__")


def get_layout_name(layout) -> str | None:
    if layout.is_record:
        return get_record_name(layout._parameters)
    else:
        return get_array_name(layout._parameters)


def get_array_class(layout, behavior: Mapping | None) -> type:
    from awkward.highlevel import Array

    behavior = overlay_behavior(behavior)

    # __array__ is a fallback for __name__. If one of these parameters is set, we should return a registered behavior
    # class, or the default array type
    for param in "__name__", "__array__":
        # Did the user specify a nominal parameter?
        name = layout.parameter(param)
        if name is None:
            continue
        # Did the user register a behavior class?
        cls = behavior.get(name)
        # Is the behavior class valid?
        if isinstance(cls, type) and issubclass(cls, Array):
            return cls
        elif cls is not None:
            raise TypeError(
                f"a non ak.Array subclass was encountered when resolving the array class for {name}"
            )

    # At this point, we just load the record array class
    if name is None:
        purelist_name = layout.purelist_parameter("__record__")
        if purelist_name is not None:
            cls = behavior.get(("*", purelist_name))
            if isinstance(cls, type) and issubclass(cls, Array):
                return cls
            elif cls is not None:
                raise TypeError(
                    f"a non ak.Array subclass was encountered when resolving the array class for {name}"
                )
    return Array


def get_record_class(layout, behavior: Mapping | None) -> type:
    from awkward.highlevel import Record

    behavior = overlay_behavior(behavior)
    rec = layout.parameter("__record__")
    if isinstance(rec, str):
        cls = behavior.get(rec)
        if isinstance(cls, type) and issubclass(cls, Record):
            return cls
    return Record


def find_record_reducer(
    reducer, parameters: JSONMapping | None, behavior: Mapping | None
):
    behavior = overlay_behavior(behavior)
    name = get_record_name(parameters)
    if name is not None:
        return behavior.get((reducer.highlevel_function(), name))


def find_custom_cast(obj, behavior: Mapping | None):
    behavior = overlay_behavior(behavior)
    for cls in type(obj).__mro__:
        fcn = behavior.get(("__cast__", cls))
        if fcn is not None:
            return fcn
    return None


def find_ufunc_generic(ufunc, layout, behavior: Mapping | None):
    nominal_type = get_layout_name(layout)
    if nominal_type is None:
        return None
    behavior = overlay_behavior(behavior)
    fcn = behavior.get((ufunc, nominal_type))
    if fcn is None:
        fcn = behavior.get((ufuncs.ufunc, nominal_type))
    return fcn


def find_ufunc(behavior: Mapping | None, signature: tuple):
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
    behavior: Mapping | None, parameters: JSONMapping | None, default: str = None
):
    behavior = overlay_behavior(behavior)
    name = get_record_name(parameters)
    if name is None:
        return default
    else:
        return behavior.get(("__typestr__", name), default)


def find_array_typestr(
    behavior: Mapping | None, parameters: JSONMapping | None, default: str = None
):
    if parameters is None:
        return default
    behavior = overlay_behavior(behavior)
    name = get_array_name(parameters)
    if name is None:
        return default
    else:
        return behavior.get(("__typestr__", name), default)


def behavior_of(*arrays, **kwargs) -> dict:
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
