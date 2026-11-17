# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np  # noqa: TID251 - CPU-only path uses ufunc reduction with where

import awkward as ak
from awkward._backends.numpy import NumpyBackend
from awkward._layout import HighLevelContext
from awkward._namedaxis import _get_named_axis


def try_reduce_dense(
    array, axis, keepdims, mask_identity, highlevel, behavior, attrs, *, mean
):
    """Reduce ordinary CPU matrices without copying valid values."""
    if type(axis) is not int or axis not in (-1, 1):
        return None

    candidate = array.layout if isinstance(array, ak.Array) else array
    if type(candidate) not in (np.ndarray, ak.contents.NumpyArray):
        return None

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")

    if (
        type(layout) is not ak.contents.NumpyArray
        or layout.backend is not NumpyBackend.instance()
        or layout._parameters
        or _get_named_axis(ctx)
        or type(layout.data) is not np.ndarray
    ):
        return None
    data = layout.data
    if (
        data.ndim != 2
        # On short rows, masked ufunc setup can cost more than the existing reducer.
        or data.shape[1] < 128
        or not data.size
        or data.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128))
        or not data.flags.c_contiguous
    ):
        return None

    valid = np.isnan(data)
    np.logical_not(valid, out=valid)
    with np.errstate(invalid="ignore", over="ignore"):
        sums = np.add.reduce(data, axis=-1, where=valid)

    if mean:
        counts = np.count_nonzero(valid, axis=-1)
        empty = counts == 0
        values = np.full(len(sums), np.nan, dtype=np.complex128)
        with np.errstate(invalid="ignore", divide="ignore"):
            np.divide(sums, counts, out=values, where=~empty)
        if mask_identity:
            index = np.arange(len(sums), dtype=np.int64)
            index[empty] = -1
            out = ak.contents.IndexedOptionArray(
                ak.index.Index64(index), ak.contents.NumpyArray(values)
            )
        else:
            out = ak.contents.NumpyArray(values)
    else:
        out = ak.contents.NumpyArray(sums)
        if mask_identity:
            out = ak.contents.ByteMaskedArray(
                ak.index.Index8(~valid.any(axis=-1)), out, valid_when=False
            )

    if keepdims:
        out = ak.contents.RegularArray(out, 1)
    return ctx.wrap(out, highlevel=highlevel)
