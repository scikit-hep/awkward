# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

# pylint: disable=wildcard-import, unused-wildcard-import
# ruff: noqa: PLE0604

import sys
import typing
from typing import *  # noqa: F403

import numpy

__all__ = list(
    {
        "ClassVar",
        "Final",
        "Self",
        "final",
        "Protocol",
        "Unpack",
        "TypeAlias",
        "TypeGuard",
        "runtime_checkable",
        "AxisMaybeNone",
        "TypedDict",
        "Literal",
        "SupportsIndex",
        "ParamSpec",
        "EllipsisType",
        *typing.__all__,
    }
)


AxisMaybeNone = TypeVar("AxisMaybeNone", int, None)  # noqa: F405

if sys.version_info < (3, 11):
    from typing import (
        ClassVar,
        Final,
        Literal,
        SupportsIndex,
        TypeAlias,
        TypeGuard,
        final,
        runtime_checkable,
    )

    from typing_extensions import (
        ParamSpec,
        Protocol,
        Self,
        TypedDict,
        Unpack,
    )

    EllipsisType = type(...)
else:
    from types import EllipsisType
    from typing import (
        ClassVar,
        Final,
        Literal,
        ParamSpec,
        Protocol,
        Self,
        SupportsIndex,
        TypeAlias,
        TypedDict,
        TypeGuard,
        Unpack,
        final,
        runtime_checkable,
    )


JSONSerializable: TypeAlias = (
    "str | int | float | bool | None | list | tuple | JSONMapping"
)
JSONMapping: TypeAlias = "dict[str, JSONSerializable]"

DType: TypeAlias = numpy.dtype


class NominalMeta(type(Protocol)):
    """Metaclass restoring the interpreter's ``isinstance`` fast path.

    Concrete classes that descend from a ``Protocol`` inherit
    ``typing._ProtocolMeta``, whose Python-level ``__instancecheck__`` costs
    roughly four times a plain class check even though, for a non-protocol
    subclass, it only performs the ordinary nominal test. Classes using this
    metaclass are checked nominally and cannot be given virtual subclasses
    through ``abc``'s ``register()``.
    """

    __instancecheck__ = type.__instancecheck__
    __subclasscheck__ = type.__subclasscheck__
