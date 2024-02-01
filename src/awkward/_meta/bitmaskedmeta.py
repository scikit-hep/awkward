# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._meta.meta import Meta
from awkward._typing import Generic, JSONSerializable, TypeVar

T = TypeVar("T", bound=Meta)


class BitMaskedMeta(Meta, Generic[T]):
    _content: T
    is_option = True

    @property
    def is_identity_like(self) -> bool:
        return False

    def purelist_parameters(self, *keys: str) -> JSONSerializable:
        if self._parameters is not None:
            for key in keys:
                if key in self._parameters:
                    return self._parameters[key]

        return self._content.purelist_parameters(*keys)

    @property
    def purelist_isregular(self) -> bool:
        return self._content.purelist_isregular

    @property
    def purelist_depth(self) -> int:
        return self._content.purelist_depth

    @property
    def minmax_depth(self) -> tuple[int, int]:
        return self._content.minmax_depth

    @property
    def branch_depth(self) -> tuple[bool, int]:
        return self._content.branch_depth

    @property
    def fields(self) -> list[str]:
        return self._content.fields

    @property
    def is_tuple(self) -> bool:
        return self._content.is_tuple

    @property
    def dimension_optiontype(self) -> bool:
        return True

    @property
    def content(self) -> T:
        return self._content
