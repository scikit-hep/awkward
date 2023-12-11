# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

# pylint: disable=wildcard-import, unused-wildcard-import
# ruff: noqa: PLE0604
from __future__ import annotations

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
        *typing.__all__,
    }
)


AxisMaybeNone = TypeVar("AxisMaybeNone", int, None)  # noqa: F405

if sys.version_info < (3, 11):
    from typing import ClassVar, Final, SupportsIndex, final, runtime_checkable

    from typing_extensions import (
        Literal,
        ParamSpec,
        Protocol,
        Self,
        TypeAlias,
        TypedDict,
        TypeGuard,
        Unpack,
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
