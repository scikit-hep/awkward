from __future__ import annotations

from awkward._typing import Protocol, Self


class PrivateSingleton(Protocol):
    _instance: Self

    @classmethod
    def _new(cls, *args, **kwargs) -> Self:
        if hasattr(cls, "_instance"):
            raise RuntimeError

        self = super().__new__(cls)
        self.__init__(*args, **kwargs)
        cls._instance = self

        return self

    def __new__(cls, *args, **kwargs):
        raise TypeError(
            f"internal_error: {cls.__name__} class should never be directly instantiated."
        )

    def __reduce__(self):
        return type(self)._reduce_constructor, ()

    @classmethod
    def _reduce_constructor(cls) -> Self:
        return cls._instance


class PublicSingleton(PrivateSingleton, Protocol):
    @classmethod
    def instance(cls, *args, **kwargs) -> Self:
        try:
            return cls._instance
        except AttributeError:
            return cls._new(*args, **kwargs)
