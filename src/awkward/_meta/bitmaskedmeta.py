# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._meta.meta import Meta, is_indexed, is_option
from awkward._parameters import type_parameters_equal
from awkward._typing import JSONSerializable


class BitMaskedMeta(Meta):
    _content: Meta
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
    def content(self) -> Meta:
        return self._content

    def _mergeable_next(self, other: Meta, mergebool: bool) -> bool:
        # Is the other content is an identity, or a union?
        if other.is_identity_like or other.is_union:
            return True
        # We can only combine option/indexed types whose array-record parameters agree
        elif is_option(other) or is_indexed(other):
            return self._content._mergeable_next(
                other.content, mergebool
            ) and type_parameters_equal(self._parameters, other._parameters)
        else:
            return self._content._mergeable_next(other, mergebool)
