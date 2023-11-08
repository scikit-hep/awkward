# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from collections.abc import Mapping

from awkward._typing import Any, JSONMapping


def attrs_of_obj(obj, attrs: Mapping | None = None) -> Mapping | None:
    from awkward.highlevel import Array, ArrayBuilder, Record

    if attrs is not None:
        return attrs
    elif isinstance(obj, (Array, Record, ArrayBuilder)):
        return obj._attrs
    else:
        return None


def attrs_of(*arrays, attrs: Mapping | None = None) -> Mapping:
    # An explicit 'attrs' always wins.
    if attrs is not None:
        return attrs

    copied = False
    for x in reversed(arrays):
        x_attrs = attrs_of_obj(x)
        if x_attrs is None:
            continue
        if attrs is None:
            attrs = x_attrs
        elif attrs is x_attrs:
            pass
        elif not copied:
            attrs = dict(attrs)
            attrs.update(x_attrs)
            copied = True
        else:
            attrs.update(x_attrs)
    return attrs


def without_transient_attrs(attrs: dict[str, Any]) -> JSONMapping:
    return {k: v for k, v in attrs.items() if not k.startswith("@")}
