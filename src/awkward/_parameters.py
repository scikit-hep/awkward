from __future__ import annotations

from collections.abc import Collection

from awkward._typing import JSONMapping, JSONSerializable


def type_parameters_equal(
    one: JSONMapping | None, two: JSONMapping | None, *, allow_missing: bool = False
) -> bool:
    if one is None and two is None:
        return True

    elif one is None:
        # NB: __categorical__ is currently a type-only parameter, but
        # we check it here as types check this too.
        for key in ("__array__", "__record__", "__categorical__"):
            if two.get(key) is not None:
                return allow_missing
        return True

    elif two is None:
        for key in ("__array__", "__record__", "__categorical__"):
            if one.get(key) is not None:
                return allow_missing
        return True

    else:
        for key in ("__array__", "__record__", "__categorical__"):
            if one.get(key) != two.get(key):
                return False
        return True


def parameters_are_equal(
    one: JSONMapping, two: JSONMapping, only_array_record=False
) -> bool:
    if one is None and two is None:
        return True
    elif one is None:
        if only_array_record:
            # NB: __categorical__ is currently a type-only parameter, but
            # we check it here as types check this too.
            for key in ("__array__", "__record__", "__categorical__"):
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
            for key in ("__array__", "__record__", "__categorical__"):
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
            keys = ("__array__", "__record__", "__categorical__")
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
    exclude: Collection[tuple[str, JSONSerializable]] = (),
) -> JSONMapping | None:
    """
    Args:
        left: first parameters mapping
        right: second parameters mapping
        exclude: collection of (key, value) items to exclude

    Returns the intersected key-value pairs of `left` and `right` as a dictionary.
    """
    if left is None or right is None:
        return None

    common_keys = iter(left.keys() & right.keys())
    has_no_exclusions = len(exclude) == 0

    # Avoid creating `result` unless we have to
    for key in common_keys:
        left_value = left[key]
        # Do our keys match?
        if (
            left_value is not None
            and left_value == right[key]
            and (has_no_exclusions or (key, left_value) not in exclude)
        ):
            # Exit, indicating that we want to create `result`
            break
    else:
        return None

    # We found a meaningful key, so create a result dict
    result = {key: left_value}
    for key in common_keys:
        left_value = left[key]
        if (
            left_value is not None
            and left_value == right[key]
            and (has_no_exclusions or (key, left_value) not in exclude)
        ):
            result[key] = left_value

    return result


def parameters_union(
    left: JSONMapping | None,
    right: JSONMapping | None,
    *,
    exclude: Collection[tuple[str, JSONSerializable]] = (),
) -> JSONMapping | None:
    """
    Args:
        left: first parameters mapping
        right: second parameters mapping
        exclude: collection of (key, value) items to exclude

    Returns the merged key-value pairs of `left` and `right` as a dictionary.

    """
    has_no_exclusions = len(exclude) == 0
    if left is None:
        if right is None:
            return None
        else:
            return {
                k: v
                for k, v in right.items()
                if v is not None and (has_no_exclusions or (k, v) not in exclude)
            }
    else:
        result = {
            k: v
            for k, v in left.items()
            if v is not None and (has_no_exclusions or (k, v) not in exclude)
        }
        if right is None:
            return result
        else:
            for key in right:
                right_value = right[key]
                if right_value is not None and (
                    has_no_exclusions or (key, right_value) not in exclude
                ):
                    result[key] = right_value

            return result


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
