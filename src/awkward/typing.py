# pylint: disable=wildcard-import, unused-wildcard-import
from __future__ import annotations

import sys
import typing
from typing import *  # noqa: F401, F403

__all__ = [*typing.__all__, "AxisMaybeNone"]


AxisMaybeNone = TypeVar("AxisMaybeNone", int, None)


if sys.version_info < (3, 11):
    import typing_extensions
    from typing_extensions import *  # noqa: F401, F403

    __all__ += typing_extensions.__all__


def __dir__() -> tuple[str, ...]:
    return __all__
