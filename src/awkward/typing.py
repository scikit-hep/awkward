import typing

if typing.TYPE_CHECKING:
    from typing_extensions import TypeAlias

    # from typing_extensions import Self
    Self = typing.Any
else:
    TypeAlias = object
    Self = object
