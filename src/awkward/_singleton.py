# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._typing import Protocol, Self


class PrivateSingleton(Protocol):
    _instance: Self

    @classmethod
    def _new(cls) -> Self:
        if hasattr(cls, "_instance"):
            raise RuntimeError(
                f"internal_error: singleton {cls.__name__} was already instantiated"
            )

        self = super().__new__(cls)
        self.__init__()  # pylint: disable=unnecessary-dunder-call
        cls._instance = self

        return self

    @classmethod
    def _ensure_instance(cls):
        try:
            return cls._instance
        except AttributeError:
            return cls._new()

    def __new__(cls, *args, **kwargs):
        raise TypeError(
            f"internal_error: {cls.__name__} class should never be directly instantiated."
        )

    def __reduce__(self):
        return type(self)._ensure_instance, ()

    @classmethod
    def _reduce_constructor(cls) -> Self:
        return cls._instance


class PublicSingleton(PrivateSingleton, Protocol):
    @classmethod
    def instance(cls) -> Self:
        return cls._ensure_instance()
