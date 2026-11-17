# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._meta.meta import Meta
from awkward._parameters import type_parameters_equal
from awkward._typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from awkward._meta.bitmaskedmeta import BitMaskedMeta
    from awkward._meta.bytemaskedmeta import ByteMaskedMeta
    from awkward._meta.indexedmeta import IndexedMeta
    from awkward._meta.indexedoptionmeta import IndexedOptionMeta
    from awkward._meta.listmeta import ListMeta
    from awkward._meta.listoffsetmeta import ListOffsetMeta
    from awkward._meta.numpymeta import NumpyMeta
    from awkward._meta.regularmeta import RegularMeta
    from awkward._meta.unmaskedmeta import UnmaskedMeta

    _Wrapper: TypeAlias = (
        IndexedMeta | IndexedOptionMeta | ByteMaskedMeta | BitMaskedMeta | UnmaskedMeta
    )
    _List: TypeAlias = RegularMeta | ListMeta | ListOffsetMeta


def mergeable_list(
    self: _List,
    other: Meta,
    mergebool: bool,
    mergecastable: Literal["same_kind", "equiv", "family"],
) -> bool:
    if other.is_identity_like or other.is_union:
        return True
    if other.is_indexed or other.is_option:
        if TYPE_CHECKING:
            assert isinstance(other, _Wrapper)
        return self._mergeable_next(other.content, mergebool, mergecastable)
    if not type_parameters_equal(self._parameters, other._parameters):
        return False
    if other.is_list:
        if TYPE_CHECKING:
            assert isinstance(other, (RegularMeta, ListMeta, ListOffsetMeta))
        return self.content._mergeable_next(other.content, mergebool, mergecastable)
    if other.is_numpy:
        if TYPE_CHECKING:
            assert isinstance(other, NumpyMeta)
        if other._mergeable_ndim > 1:
            return self._mergeable_next(
                other._mergeable_regular(), mergebool, mergecastable
            )
    return False


def mergeable_wrapper(
    self: _Wrapper,
    other: Meta,
    mergebool: bool,
    mergecastable: Literal["same_kind", "equiv", "family"],
) -> bool:
    if other.is_identity_like or other.is_union:
        return True
    if other.is_indexed or other.is_option:
        if TYPE_CHECKING:
            assert isinstance(
                other,
                (
                    IndexedMeta,
                    IndexedOptionMeta,
                    ByteMaskedMeta,
                    BitMaskedMeta,
                    UnmaskedMeta,
                ),
            )
        return self.content._mergeable_next(other.content, mergebool, mergecastable)
    return self.content._mergeable_next(other, mergebool, mergecastable)
