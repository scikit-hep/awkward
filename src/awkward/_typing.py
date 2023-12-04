# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

# pylint: disable=wildcard-import, unused-wildcard-import
# ruff: noqa: PLE0604
from __future__ import annotations

import sys
import typing
from typing import *  # noqa: F403
from typing import TypeVar

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
        *typing.__all__,
    }
)

if sys.version_info < (3, 11):
    from typing import ClassVar, Final, SupportsIndex, runtime_checkable

    from typing_extensions import (
        Literal,
        ParamSpec,
        Protocol,
        Self,
        TypeAlias,
        TypedDict,
        TypeGuard,
        Unpack,
        final,
    )
else:
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


AxisMaybeNone = TypeVar("AxisMaybeNone", int, None)


T = TypeVar("T")


class ImplementsReadOnlyProperty(Protocol[T]):
    def __get__(self, instance, owner=None) -> T:
        ...
