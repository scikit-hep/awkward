# pylint: disable=wildcard-import, unused-wildcard-import
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
        *typing.__all__,
    }
)


AxisMaybeNone = TypeVar("AxisMaybeNone", int, None)


if sys.version_info < (3, 11):
    from typing_extensions import (  # noqa: F401, F403
        Final,
        Literal,
        Protocol,
        Self,
        TypeAlias,
        TypedDict,
        Unpack,
        final,
        runtime_checkable,
    )
else:
    from typing import (  # noqa: F401, F403
        Final,
        Literal,
        Protocol,
        Self,
        TypeAlias,
        TypedDict,
        Unpack,
        final,
        runtime_checkable,
    )
