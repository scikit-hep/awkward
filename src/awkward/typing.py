# pylint: disable=wildcard-import, unused-wildcard-import
# ruff: noqa: PLE0604
from __future__ import annotations

import sys
import typing
from typing import *  # noqa: F403

__all__ = list(
    {
        "Final",
        "Self",
        "final",
        "Protocol",
        "Unpack",
        "TypeAlias",
        "runtime_checkable",
        "AxisMaybeNone",
        "TypedDict",
        "Literal",
        "SupportsIndex",
        *typing.__all__,
    }
)


AxisMaybeNone = TypeVar("AxisMaybeNone", int, None)  # noqa: F405

if sys.version_info < (3, 11):
    from typing_extensions import (
        Final,
        Literal,
        Protocol,
        Self,
        SupportsIndex,
        TypeAlias,
        TypedDict,
        Unpack,
        final,
        runtime_checkable,
    )
else:
    from typing import (
        Final,
        Literal,
        Protocol,
        Self,
        SupportsIndex,
        TypeAlias,
        TypedDict,
        Unpack,
        final,
        runtime_checkable,
    )