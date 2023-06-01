# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from awkward._typing import Self, TypeAlias

ShapeItem: TypeAlias = "int | type[unknown_length]"


class _UnknownLength:
    _name: str

    @classmethod
    def _new(cls, name: str) -> Self:
        self = super().__new__(cls)
        self._name = name
        return self

    def __new__(cls, *args, **kwargs):
        raise TypeError(
            "internal_error: the _UnknownLength class should never be directly instantiated"
        )

    def __add__(self, other) -> Self | NotImplemented:
        if isinstance(other, int) or other is self:
            return self
        else:
            return NotImplemented

    __radd__ = __add__
    __iadd__ = __add__

    def __sub__(self, other) -> Self | NotImplemented:
        if isinstance(other, int) or other is self:
            return self
        else:
            return NotImplemented

    __rsub__ = __sub__
    __isub__ = __sub__

    def __mul__(self, other) -> Self | NotImplemented:
        if isinstance(other, int) or other is self:
            return self
        else:
            return NotImplemented

    __rmul__ = __mul__
    __imul__ = __mul__

    def __floordiv__(self, other) -> Self | NotImplemented:
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
        return f"{__name__}.{self._name}"

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


unknown_length = _UnknownLength._new("unknown_length")
