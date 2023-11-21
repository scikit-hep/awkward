# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._typing import TYPE_CHECKING, TypeGuard, TypeVar

if TYPE_CHECKING:
    from awkward._meta.bitmaskedmeta import BitMaskedMeta
    from awkward._meta.bytemaskedmeta import ByteMaskedMeta
    from awkward._meta.indexedmeta import IndexedMeta
    from awkward._meta.indexedoptionmeta import IndexedOptionMeta
    from awkward._meta.listmeta import ListMeta
    from awkward._meta.listoffsetmeta import ListOffsetMeta
    from awkward._meta.meta import Meta
    from awkward._meta.numpymeta import NumpyMeta
    from awkward._meta.recordmeta import RecordMeta
    from awkward._meta.regularmeta import RegularMeta
    from awkward._meta.unionmeta import UnionMeta
    from awkward._meta.unmaskedmeta import UnmaskedMeta

np = NumpyMetadata.instance()


T = TypeVar("T", bound="Meta")


def is_option(
    meta: Meta
) -> TypeGuard[IndexedOptionMeta | BitMaskedMeta | ByteMaskedMeta | UnmaskedMeta]:
    return meta.is_option


def is_list(meta: Meta) -> TypeGuard[RegularMeta | ListOffsetMeta | ListMeta]:
    return meta.is_list


def is_numpy(meta: Meta) -> TypeGuard[NumpyMeta]:
    return meta.is_numpy


def is_regular(meta: Meta) -> TypeGuard[RegularMeta]:
    return meta.is_regular


def is_union(meta: Meta) -> TypeGuard[UnionMeta]:
    return meta.is_union


def is_record(meta: Meta) -> TypeGuard[RecordMeta]:
    return meta.is_record


def is_indexed(meta: Meta) -> TypeGuard[IndexedOptionMeta, IndexedMeta]:
    return meta.is_indexed


class ImplementsTuple(RecordMeta):  # Intersection
    _fields: None


def is_record_tuple(meta: Meta) -> TypeGuard[ImplementsTuple]:
    return meta.is_record and meta.is_tuple


class ImplementsRecord(RecordMeta):
    _fields: list[str]


def is_record_record(meta: Meta) -> TypeGuard[ImplementsRecord]:
    return meta.is_record and not meta.is_tuple


def mergeable(one: Meta, two: Meta, mergebool: bool = True) -> bool:
    return one._mergeable_next(two, mergebool=mergebool)
