# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward import forms
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._parameters import type_parameters_equal
from awkward._typing import Literal
from awkward.types.numpytype import primitive_to_dtype

np = NumpyMetadata.instance()
numpy = Numpy.instance()


def mergeable(
    one: forms.Form,
    two: forms.Form,
    mergebool: bool,
    mergecastable: Literal["same_kind", "equiv", "family"],
) -> bool:
    """The Content mergeability rules, evaluated without constructing buffers.

    This is compatibility for merging, not form equality: unions can always
    absorb another form, and regular lists of different sizes can become jagged.
    Keep this predicate in agreement with Content._mergeable_next.
    """
    if isinstance(one, (forms.EmptyForm, forms.UnionForm)):
        return True
    if two.is_identity_like or two.is_union:
        return True

    wrappers = (
        forms.IndexedForm,
        forms.IndexedOptionForm,
        forms.ByteMaskedForm,
        forms.BitMaskedForm,
        forms.UnmaskedForm,
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

    if isinstance(one, forms.NumpyForm):
        if one.inner_shape:
            return mergeable(one.to_RegularForm(), two, mergebool, mergecastable)
        if not isinstance(two, forms.NumpyForm) or two.inner_shape:
            return False
        left = primitive_to_dtype(one.primitive)
        right = primitive_to_dtype(two.primitive)
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

    lists = (forms.RegularForm, forms.ListForm, forms.ListOffsetForm)
    if isinstance(one, lists):
        if isinstance(two, lists):
            return mergeable(one.content, two.content, mergebool, mergecastable)
        if isinstance(two, forms.NumpyForm) and two.inner_shape:
            return mergeable(one, two.to_RegularForm(), mergebool, mergecastable)
        return False

    if isinstance(one, forms.RecordForm) and isinstance(two, forms.RecordForm):
        if one.is_tuple != two.is_tuple or set(one.fields) != set(two.fields):
            return False
        return all(
            mergeable(one.content(field), two.content(field), mergebool, mergecastable)
            for field in one.fields
        )
    return False
