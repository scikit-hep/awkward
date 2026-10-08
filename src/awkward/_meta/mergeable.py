# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._meta.bitmaskedmeta import BitMaskedMeta
from awkward._meta.bytemaskedmeta import ByteMaskedMeta
from awkward._meta.emptymeta import EmptyMeta
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
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._parameters import type_parameters_equal
from awkward._typing import Literal

np = NumpyMetadata.instance()
numpy = Numpy.instance()


def mergeable(
    one: Meta,
    two: Meta,
    mergebool: bool,
    mergecastable: Literal["same_kind", "equiv", "family"],
) -> bool:
    """Shared Form/Content mergeability rules, evaluated without constructing buffers.

    This is compatibility for merging, not form equality: unions can always
    absorb another form, and regular lists of different sizes can become jagged.
    Data-dependent merging is implemented separately by Content.
    """
    if isinstance(one, (EmptyMeta, UnionMeta)):
        return True
    if two.is_identity_like or two.is_union:
        return True

    wrappers = (
        IndexedMeta,
        IndexedOptionMeta,
        ByteMaskedMeta,
        BitMaskedMeta,
        UnmaskedMeta,
    )
    if isinstance(one, wrappers):
        return mergeable(
            one.content,
            two.content if isinstance(two, wrappers) else two,
            mergebool,
            mergecastable,
        )
    if isinstance(two, wrappers):
        return mergeable(one, two.content, mergebool, mergecastable)
    if not type_parameters_equal(one._parameters, two._parameters):
        return False

    if isinstance(one, NumpyMeta):
        if one._mergeable_ndim > 1:
            return mergeable(one._mergeable_regular(), two, mergebool, mergecastable)
        if not isinstance(two, NumpyMeta) or two._mergeable_ndim > 1:
            return False
        left = one.dtype
        right = two.dtype
        if left == right:
            return True
        if (np.issubdtype(left, np.bool_) and np.issubdtype(right, np.number)) or (
            np.issubdtype(left, np.number) and np.issubdtype(right, np.bool_)
        ):
            return mergebool
        if any(
            np.issubdtype(dtype, temporal)
            for dtype in (left, right)
            for temporal in (np.datetime64, np.timedelta64)
        ):
            return False
        if mergecastable == "family":
            return any(
                np.issubdtype(left, family) and np.issubdtype(right, family)
                for family in (np.integer, np.floating, np.complexfloating)
            )
        if mergecastable in ("same_kind", "equiv"):
            return numpy.can_cast(left, right, casting=mergecastable) or numpy.can_cast(
                right, left, casting=mergecastable
            )
        raise TypeError(f"unrecognized mergecastable option: {mergecastable}")

    lists = (RegularMeta, ListMeta, ListOffsetMeta)
    if isinstance(one, lists):
        if isinstance(two, lists):
            return mergeable(one.content, two.content, mergebool, mergecastable)
        if isinstance(two, NumpyMeta) and two._mergeable_ndim > 1:
            return mergeable(one, two._mergeable_regular(), mergebool, mergecastable)
        return False

    if isinstance(one, RecordMeta) and isinstance(two, RecordMeta):
        if one.is_tuple != two.is_tuple or set(one.fields) != set(two.fields):
            return False
        return all(
            mergeable(content, two.content(field), mergebool, mergecastable)
            for field, content in zip(one.fields, one.contents, strict=True)
        )
    return False
