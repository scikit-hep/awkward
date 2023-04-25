from __future__ import annotations

from awkward._typing import Protocol, Self


class Singleton(Protocol):
    _instance: type[Self]

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def instance(cls) -> Self:
        try:
            return cls._instance
        except AttributeError:
            cls._instance = cls()
            return cls._instance
