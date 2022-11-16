# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import annotations

from typing import Union

from awkward import _errors


class UnknownValue:
    def __repr__(self):
        return "UnknownLength"

    def __str__(self):
        return "??"

    def __eq__(self, other):
        return unknown_value

    def __add__(self, other):
        return unknown_value

    def __radd__(self, other):
        return unknown_value

    def __sub__(self, other):
        return unknown_value

    def __rsub__(self, other):
        return unknown_value

    def __mul__(self, other):
        return unknown_value

    def __rmul__(self, other):
        return unknown_value

    def __truediv__(self, other):
        return unknown_value

    def __floordiv__(self, other):
        return unknown_value

    def __rdiv__(self, other):
        return unknown_value

    def __rfloordiv__(self, other):
        return unknown_value

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return False

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return False

    def __bool__(self):
        raise _errors.wrap_error(ValueError("cannot realise an unknown value"))

    def __int__(self):
        raise _errors.wrap_error(ValueError("cannot realise an unknown value"))

    def __index__(self):
        raise _errors.wrap_error(ValueError("cannot realise an unknown value"))


unknown_value: UnknownValue = UnknownValue()

ShapeItem = Union[int, UnknownValue]
Shape = tuple[ShapeItem, ...]


def shapes_are_compatible(left: Shape, right: Shape) -> bool:
    if len(left) != len(right):
        return False

    for this, that in zip(left, right):
        components_are_equal = this == that
        if components_are_equal is unknown_value:
            continue
        if not components_are_equal:
            return False
    return True
