import typing

if typing.TYPE_CHECKING:
    from typing_extensions import Self, TypeAlias
else:
    TypeAlias = object
    Self = object
