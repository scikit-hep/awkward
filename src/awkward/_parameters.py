# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Collection
from itertools import chain

from awkward._typing import JSONMapping, JSONSerializable

TYPE_PARAMETERS = ("__array__", "__list__", "__record__", "__categorical__")


def type_parameters_equal(
    one: JSONMapping | None, two: JSONMapping | None, *, allow_missing: bool = False
) -> bool:
    if one is None and two is None:
        return True

    elif one is None:
        # NB: __categorical__ is currently a type-only parameter, but
        # we check it here as types check this too.
        for key in TYPE_PARAMETERS:
            if two.get(key) is not None:
                return allow_missing
        return True

    elif two is None:
        for key in TYPE_PARAMETERS:
            if one.get(key) is not None:
                return allow_missing
        return True

    else:
        for key in TYPE_PARAMETERS:
            if one.get(key) != two.get(key):
                return False
        return True


def parameters_are_equal(
    one: JSONMapping | None, two: JSONMapping | None, only_array_record=False
) -> bool:
    if one is None and two is None:
        return True
    elif one is None:
        if only_array_record:
            # NB: __categorical__ is currently a type-only parameter, but
            # we check it here as types check this too.
            for key in TYPE_PARAMETERS:
                if two.get(key) is not None:
                    return False
            return True
        else:
            for value in two.values():
                if value is not None:
                    return False
            return True

    elif two is None:
        if only_array_record:
            for key in TYPE_PARAMETERS:
                if one.get(key) is not None:
                    return False
            return True
        else:
            for value in one.values():
                if value is not None:
                    return False
            return True

    else:
        if only_array_record:
            keys = TYPE_PARAMETERS
        else:
            keys = set(one.keys()).union(two.keys())
        for key in keys:
            if one.get(key) != two.get(key):
                return False
        return True


def parameters_intersect(
    left: JSONMapping | None,
    right: JSONMapping | None,
    *,
    exclude: Collection[tuple[str, JSONSerializable]] | None = None,
) -> JSONMapping | None:
    """
    Args:
        left: first parameters mapping
        right: second parameters mapping
        exclude: collection of keys to exclude

    Returns the intersected key-value pairs of `left` and `right` as a dictionary.
    """
    if left is None or right is None:
        return None

    common_keys = iter(left.keys() & right.keys())
    has_exclusions = exclude is not None and len(exclude) > 0

    # Avoid creating `result` unless we have to
    result = None
    for key in common_keys:
        left_value = left[key]
        # Do our keys match?
        if (
            left_value is not None
            and left_value == right[key]
            and not (
                has_exclusions and any(pair == (key, left_value) for pair in exclude)
            )
        ):
            # Exit, indicating that we want to create `result`
            if result is None:
                result = {key: left_value}
            else:
                result[key] = left_value
    return result


def parameters_union(
    left: JSONMapping | None,
    right: JSONMapping | None,
    *,
    exclude: Collection[tuple[str, JSONSerializable]] | None = None,
) -> JSONMapping | None:
    """
    Args:
        left: first parameters mapping
        right: second parameters mapping
        exclude: collection of key items to exclude

    Returns the merged key-value pairs of `left` and `right` as a dictionary.

    """
    has_exclusions = exclude is not None and len(exclude) > 0
    items = []
    if left is not None:
        items.append(left.items())
    if right is not None:
        items.append(right.items())

    parameters = None
    for item in chain.from_iterable(items):
        key, value = item
        if value is None:
            continue
        if has_exclusions and any(pair == item for pair in exclude):
            continue
        if parameters is None:
            parameters = {key: value}
        else:
            parameters[key] = value

    return parameters


def parameters_are_empty(parameters: JSONMapping | None) -> bool:
    """
    Args:
        parameters (dict or None): parameters dictionary, or None

    Return True if the parameters dictionary is considered empty, either because it is
    None, or because it does not have any meaningful (non-None) values; otherwise,
    return False.
    """
    if parameters is None:
        return True

    for item in parameters.values():
        if item is not None:
            return False

    return True
