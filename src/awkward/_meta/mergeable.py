# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._meta.meta import Meta
from awkward._meta.recordmeta import RecordMeta
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._parameters import type_parameters_equal
from awkward._typing import TYPE_CHECKING, Literal

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

    _Wrappers = (
        IndexedMeta,
        IndexedOptionMeta,
        ByteMaskedMeta,
        BitMaskedMeta,
        UnmaskedMeta,
    )
    _Lists = (RegularMeta, ListMeta, ListOffsetMeta)

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
    if one.is_unknown or one.is_union:
        return True
    if two.is_identity_like or two.is_union:
        return True

    if one.is_indexed or one.is_option:
        if TYPE_CHECKING:
            assert isinstance(one, _Wrappers)
        if two.is_indexed or two.is_option:
            if TYPE_CHECKING:
                assert isinstance(two, _Wrappers)
            return mergeable(one.content, two.content, mergebool, mergecastable)
        return mergeable(one.content, two, mergebool, mergecastable)
    if two.is_indexed or two.is_option:
        if TYPE_CHECKING:
            assert isinstance(two, _Wrappers)
        return mergeable(one, two.content, mergebool, mergecastable)
    if not type_parameters_equal(one._parameters, two._parameters):
        return False

    if one.is_numpy:
        if TYPE_CHECKING:
            assert isinstance(one, NumpyMeta)
        if one._mergeable_ndim > 1:
            return mergeable(one._mergeable_regular(), two, mergebool, mergecastable)
        if not two.is_numpy:
            return False
        if TYPE_CHECKING:
            assert isinstance(two, NumpyMeta)
        if two._mergeable_ndim > 1:
            return False
        left = one._mergeable_dtype
        right = two._mergeable_dtype
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
            # Casting compares dtype descriptors, never device buffers. Use the
            # same NumPy rule for forms, layouts and mixed-backend metadata.
            return numpy.can_cast(left, right, casting=mergecastable) or numpy.can_cast(
                right, left, casting=mergecastable
            )
        raise TypeError(f"unrecognized mergecastable option: {mergecastable}")

    if one.is_list:
        if TYPE_CHECKING:
            assert isinstance(one, _Lists)
        if two.is_list:
            if TYPE_CHECKING:
                assert isinstance(two, _Lists)
            return mergeable(one.content, two.content, mergebool, mergecastable)
        if two.is_numpy:
            if TYPE_CHECKING:
                assert isinstance(two, NumpyMeta)
            if two._mergeable_ndim > 1:
                return mergeable(
                    one, two._mergeable_regular(), mergebool, mergecastable
                )
        return False

    if one.is_record and two.is_record:
        if TYPE_CHECKING:
            assert isinstance(one, RecordMeta) and isinstance(two, RecordMeta)
        if one.is_tuple != two.is_tuple:
            return False
        left_fields, right_fields = one.fields, two.fields
        if left_fields == right_fields:
            # Raw child metadata avoids RecordArray.content's length resolution
            # and slicing. Matching field order also needs no name lookup.
            return all(
                mergeable(left, right, mergebool, mergecastable)
                for left, right in zip(one.contents, two.contents, strict=True)
            )
        if set(left_fields) != set(right_fields):
            return False
        return all(
            mergeable(content, RecordMeta.content(two, field), mergebool, mergecastable)
            for field, content in zip(left_fields, one.contents, strict=True)
        )
    return False
