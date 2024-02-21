# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections import ChainMap
from collections.abc import Callable, Mapping

import awkward as ak
from awkward._nplikes import ufuncs
from awkward._typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from awkward._nplikes.numpy_like import UfuncLike
    from awkward._reducers import Reducer
    from awkward.contents.content import Content
    from awkward.highlevel import Array
    from awkward.highlevel import Record as HighLevelRecord
    from awkward.record import Record


def overlay_behavior(behavior: Mapping | None) -> Mapping:
    """
    Args:
        behavior: behavior dictionary, or None

    Return a ChainMap object that overlays the given behavior
    on top of the global #ak.behavior
    """
    if behavior is None:
        return ak.behavior
    else:
        return ChainMap(behavior, ak.behavior)  # type: ignore[arg-type]


def get_array_class(layout: Content, behavior: Mapping | None) -> type[Array]:
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


def get_record_class(layout: Record, behavior: Mapping | None) -> type[HighLevelRecord]:
    from awkward.highlevel import Record

    behavior = overlay_behavior(behavior)
    rec = layout.parameter("__record__")
    if isinstance(rec, str):
        cls = behavior.get(rec)
        if isinstance(cls, type) and issubclass(cls, Record):
            return cls
    return Record


def find_record_reducer(
    reducer: Reducer, layout: Record, behavior: Mapping | None
) -> Callable[[Array, bool], Any] | None:
    behavior = overlay_behavior(behavior)
    rec = layout.parameter("__record__")
    if isinstance(rec, str):
        return behavior.get((reducer.highlevel_function(), rec))
    else:
        return None


def find_custom_cast(
    obj: Any, behavior: Mapping | None
) -> Callable[[Any], Content | Record] | None:
    behavior = overlay_behavior(behavior)
    for cls in type(obj).__mro__:
        fcn = behavior.get(("__cast__", cls))
        if fcn is not None:
            return fcn
    return None


def find_ufunc_generic(
    ufunc: UfuncLike, layout: Content, behavior: Mapping | None
) -> Callable[[UfuncLike, str, list, dict], Any] | None:
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


def find_ufunc(behavior: Mapping | None, signature: tuple) -> UfuncLike | None:
    if any(s is None for s in signature):
        return None

    behavior = overlay_behavior(behavior)

    # Try and fast-path the lookup
    try:
        return behavior[signature]
    except KeyError:
        # We didn't find an exact overload, and we won't find any!
        if all(isinstance(x, str) for x in signature):
            return None

    # Fall back on linear search (first-wins)
    for key, custom in behavior.items():
        if (
            isinstance(key, tuple)
            and len(key) == len(signature)
            and key[0] == signature[0]
            and all(
                k == s
                or (isinstance(k, type) and isinstance(s, type) and issubclass(s, k))
                for k, s in zip(key[1:], signature[1:])
            )
        ):
            return custom
    return None


def find_record_typestr(
    behavior: None | Mapping,
    parameters: None | Mapping[str, Any],
    default: str | None = None,
) -> str | None:
    if parameters is None:
        return default
    behavior = overlay_behavior(behavior)
    return behavior.get(("__typestr__", parameters.get("__record__")), default)


def find_array_typestr(
    behavior: None | Mapping,
    parameters: None | Mapping[str, Any],
    default: str | None = None,
) -> str | None:
    if parameters is None:
        return default
    behavior = overlay_behavior(behavior)
    return behavior.get(("__typestr__", parameters.get("__list__")), default)


def behavior_of_obj(obj: Any, behavior: Mapping | None = None) -> Mapping | None:
    from awkward.highlevel import Array, ArrayBuilder, Record

    if behavior is not None:
        return behavior
    elif isinstance(obj, (Array, Record, ArrayBuilder)):
        return obj._behavior
    else:
        return None


def behavior_of(*arrays: Any, behavior: Mapping | None = None) -> Mapping | None:
    if behavior is not None:
        # An explicit 'behavior' always wins.
        return behavior

    copied = False
    for x in arrays[::-1]:
        x_behavior = behavior_of_obj(x)
        # Don't merge shared behaviors!
        if x_behavior is None or behavior is x_behavior:
            pass
        elif behavior is None:
            behavior = x_behavior
        elif not copied:
            behavior = dict(behavior)
            behavior.update(x_behavior)
            copied = True
        else:
            assert isinstance(behavior, dict)
            behavior.update(x_behavior)
    return behavior
