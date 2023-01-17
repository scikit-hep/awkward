from __future__ import annotations

from awkward.typing import Protocol, Self


class Singleton(Protocol):
    _instance: type[Self]

    @classmethod
    def instance(cls) -> Self:
        try:
            return cls._instance
        except AttributeError:
            cls._instance = cls()
            return cls._instance
