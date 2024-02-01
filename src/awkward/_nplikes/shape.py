# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._singleton import PrivateSingleton
from awkward._typing import TYPE_CHECKING, Self, TypeAlias

__all__ = ("ShapeItem", "UnknownLength", "unknown_length")

ShapeItem: TypeAlias = "int | UnknownLength"

if TYPE_CHECKING:
    from types import NotImplementedType


class UnknownLength(PrivateSingleton):
    _instance_name: str

    def __add__(self, other) -> Self | NotImplementedType:
        if isinstance(other, int) or other is self:
            return self
        else:
            return NotImplemented

    __radd__ = __add__
    __iadd__ = __add__

    def __sub__(self, other) -> Self | NotImplementedType:
        if isinstance(other, int) or other is self:
            return self
        else:
            return NotImplemented

    __rsub__ = __sub__
    __isub__ = __sub__

    def __mul__(self, other) -> Self | NotImplementedType:
        if isinstance(other, int) or other is self:
            return self
        else:
            return NotImplemented

    __rmul__ = __mul__
    __imul__ = __mul__

    def __floordiv__(self, other) -> Self | NotImplementedType:
        if isinstance(other, int) or other is self:
            return self
        else:
            return NotImplemented

    __rfloordiv__ = __floordiv__
    __ifloordiv__ = __floordiv__

    def __divmod__(self, other) -> tuple[Self, Self]:
        return self, self

    __rdivmod__ = __divmod__

    def __str__(self) -> str:
        return "##"

    def __repr__(self):
        return self._instance_name

    def __eq__(self, other) -> bool:
        if other is self:
            return True
        else:
            raise TypeError("cannot compare unknown lengths against known values")

    def __gt__(self, other):
        raise TypeError("cannot order unknown lengths")

    def __index__(self):
        raise TypeError("cannot interpret unknown lengths as concrete index values")

    def __int__(self):
        raise TypeError("cannot interpret unknown lengths as concrete values")

    __bool__ = __int__
    __float__ = __int__

    __ge__ = __gt__
    __le__ = __gt__
    __lt__ = __gt__


# Inform the singleton if its module name
UnknownLength._instance_name = f"{__name__}.unknown_length"

# Ensure we have a single instance
unknown_length = UnknownLength._new()
