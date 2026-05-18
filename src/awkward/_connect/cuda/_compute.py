# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from cuda.compute import (
    CountingIterator,
    DiscardIterator,
    gpu_struct,
    inclusive_scan,
    reduce_into,
    OpKind,
    segmented_reduce,
    unary_transform,
)
from numba import cuda

from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy

cupy_nplike = Cupy.instance()
cp = cupy_nplike._module

numpy_nplike = Numpy.instance()
np = numpy_nplike._module

# Cache for cuda.compute availability
_cuda_compute_available: bool | None = None


def is_available() -> bool:
    global _cuda_compute_available

    if _cuda_compute_available is not None:
        return _cuda_compute_available

    try:
        import cuda.compute  # noqa: F401

        _cuda_compute_available = True
    except ImportError:
        _cuda_compute_available = False

    return _cuda_compute_available


def make_segment_views(offsets):
    """
    Returns (starts, stops) views for segmented operations.
    """
    return offsets[:-1], offsets[1:]


def normalize_index_dtype(dtype):
    dt = cp.dtype(dtype)
    if dt.kind in ("u", "i"):
        return cp.int64
    return dt


def infer_complex_dtype(dtype):
    dt = cp.dtype(dtype)
    if dt == cp.float32:
        return cp.complex64
    if dt == cp.float64:
        return cp.complex128
    raise TypeError(f"Expected float32/float64 interleaved complex buffer, got {dt}")


def segmented_sort(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable,
):
    from cuda.compute import SortOrder, segmented_sort

    cupy_nplike = Cupy.instance()
    cp = cupy_nplike._module

    # Ensure offsets are int64 as expected by segmented_sort
    if offsets.dtype != cp.int64:
        offsets = offsets.astype(cp.int64)

    num_segments = offsetslength - 1
    num_items = int(offsets[-1]) if len(offsets) > 0 else 0

    start_o, end_o = make_segment_views(offsets)

    order = SortOrder.ASCENDING if ascending else SortOrder.DESCENDING

    segmented_sort(
        d_in_keys=fromptr,
        d_out_keys=toptr,
        d_in_values=None,
        d_out_values=None,
        num_items=num_items,
        num_segments=num_segments,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        order=order,
        stream=None,
    )


def awkward_reduce_argmax(
    result,
    input_data,
    offsets_data,
    starts,
    outlength,
):
    index_dtype = normalize_index_dtype(offsets_data.dtype)

    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_argmax(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]
        if start_idx == end_idx:
            return -1
        # return a global index
        return np.argmax(segment) + start_idx

    segment_ids = CountingIterator(index_dtype(0))

    # TODO: replace with segmented_reduce once available/fixed in CCCL
    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_argmax,
        num_items=outlength,
    )


@cuda.jit(device=True)
def lexargmax_complex(values, start_idx, end_idx):
    if start_idx == end_idx:
        return np.int64(-1)

    best_idx = start_idx
    best = values[start_idx]

    i = start_idx + 1
    while i < end_idx:
        value = values[i]
        if value.real > best.real or (
            value.real == best.real and value.imag > best.imag
        ):
            best = value
            best_idx = i
        i += 1

    return best_idx


def awkward_reduce_argmax_complex(
    result,
    input_data,
    offsets_data,
    outlength,
):
    complex_dtype = infer_complex_dtype(input_data.dtype)
    input_complex = input_data.view(complex_dtype)

    index_dtype = normalize_index_dtype(offsets_data.dtype)
    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_argmax(segment_id):
        return lexargmax_complex(
            input_complex,
            start_o[segment_id],
            end_o[segment_id],
        )

    segment_ids = CountingIterator(index_dtype(0))

    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_argmax,
        num_items=outlength,
    )


def awkward_reduce_argmin(
    result,
    input_data,
    offsets_data,
    starts,
    outlength,
):
    index_dtype = normalize_index_dtype(offsets_data.dtype)
    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_argmin(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]
        if start_idx == end_idx:
            return -1
        # return a global index
        return np.argmin(segment) + start_idx

    segment_ids = CountingIterator(index_dtype(0))

    # TODO: replace with segmented_reduce once available/fixed in CCCL
    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_argmin,
        num_items=outlength,
    )


@cuda.jit(device=True)
def lexargmin_complex(values, start_idx, end_idx):
    if start_idx == end_idx:
        return np.int64(-1)

    best_idx = start_idx
    best = values[start_idx]

    i = start_idx + 1
    while i < end_idx:
        value = values[i]

        if value.real < best.real or (
            value.real == best.real and value.imag < best.imag
        ):
            best = value
            best_idx = i
        i += 1

    return best_idx


def awkward_reduce_argmin_complex(
    result,
    input_data,
    offsets_data,
    outlength,
):
    index_dtype = normalize_index_dtype(offsets_data.dtype)

    complex_dtype = infer_complex_dtype(input_data.dtype)
    input_complex = input_data.view(complex_dtype)

    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_argmin(segment_id):
        return lexargmin_complex(
            input_complex,
            start_o[segment_id],
            end_o[segment_id],
        )

    segment_ids = CountingIterator(index_dtype(0))

    unary_transform(
        d_in=segment_ids,
        d_out=result,
        op=segment_reduce_argmin,
        num_items=outlength,
    )


def awkward_reduce_sum(
    result,
    input_data,
    offsets_data,
    outlength,
):
    d_input = input_data.astype(result.dtype, copy=False)
    start_o, end_o = make_segment_views(offsets_data)

    h_init = np.asarray(0, dtype=result.dtype)

    segmented_reduce(
        d_in=d_input,
        d_out=result,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=OpKind.PLUS,
        h_init=h_init,
    )


def awkward_reduce_sum_bool(
    result,
    input_data,
    offsets_data,
    outlength,
):
    # ak.any semantics: result is True for a bin iff any element is non-zero.
    # If we feed wider-than-bool input straight into a MAX reduction and then
    # truncate to int8 on store, an input like [256, 512, ...] reduces to MAX
    # = 512 which truncates to 0 → spurious False. Map to {0, 1} first, then
    # MAX = OR. Symmetric to awkward_reduce_prod_bool (MIN = AND).
    if input_data.dtype == cp.bool_:
        mapped = input_data.view(cp.int8)
    else:
        mapped = cp.empty(input_data.shape, dtype=cp.int8)

        def is_nonzero(x):
            return cp.int8(1) if x != 0 else cp.int8(0)

        unary_transform(
            d_in=input_data,
            d_out=mapped,
            op=is_nonzero,
            num_items=input_data.size,
        )

    d_out = result.view(cp.int8) if result.dtype == cp.bool_ else result
    start_o, end_o = make_segment_views(offsets_data)
    h_init = np.asarray(0, dtype=cp.int8)  # identity for MAX over {0, 1}

    segmented_reduce(
        d_in=mapped,
        d_out=d_out,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=OpKind.MAXIMUM,
        h_init=h_init,
    )


def awkward_reduce_sum_bool_complex(
    result,
    input_data,
    offsets_data,
    outlength,
):
    complex_dtype = infer_complex_dtype(input_data.dtype)
    input_complex = input_data.view(complex_dtype)

    d_out = result.view(cp.int8) if result.dtype == cp.bool_ else result

    mapped_data = cp.empty(input_complex.shape, dtype=cp.int8)

    def is_nonzero_complex(c):
        # A complex number is non-zero if either real or imag is non-zero
        if c.real != 0 or c.imag != 0:
            return 1
        return 0

    unary_transform(
        d_in=input_complex,
        d_out=mapped_data,
        op=is_nonzero_complex,
        num_items=input_complex.size,
    )

    start_o, end_o = make_segment_views(offsets_data)
    h_init = np.asarray(0, dtype=cp.int8)  # Identity for OR is False

    segmented_reduce(
        d_in=mapped_data,
        d_out=d_out,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=OpKind.MAXIMUM,
        h_init=h_init,
    )


def awkward_reduce_sum_int32_bool_64(
    result,
    input_data,
    offsets_data,
    outlength,
):
    d_input = input_data.astype(result.dtype, copy=False)
    start_o, end_o = make_segment_views(offsets_data)
    h_init = np.asarray(0, dtype=result.dtype)

    segmented_reduce(
        d_in=d_input,
        d_out=result,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=OpKind.PLUS,
        h_init=h_init,
    )


def awkward_reduce_sum_int64_bool_64(
    result,
    input_data,
    offsets_data,
    outlength,
):
    d_input = input_data.astype(result.dtype, copy=False)
    start_o, end_o = make_segment_views(offsets_data)
    h_init = np.asarray(0, dtype=result.dtype)

    segmented_reduce(
        d_in=d_input,
        d_out=result,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=OpKind.PLUS,
        h_init=h_init,
    )


def awkward_reduce_sum_complex(
    result,
    input_data,
    offsets_data,
    outlength,
):
    complex_dtype = infer_complex_dtype(input_data.dtype)
    input_complex = input_data.view(complex_dtype)
    result_complex = result.view(complex_dtype)

    start_o, end_o = make_segment_views(offsets_data)

    h_init = np.asarray(0, dtype=complex_dtype)

    def sum_op(a, b):
        return a + b

    segmented_reduce(
        d_in=input_complex,
        d_out=result_complex,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=sum_op,
        h_init=h_init,
    )


# `awkward_reduce_sum_bool_complex` (above) handles both float32- and
# float64-interleaved inputs via `infer_complex_dtype`. The previous
# `awkward_reduce_sum_bool_complex64_64` / `_complex128_64` specialisations
# were verbatim duplicates with a hardcoded complex dtype — dispatch now
# routes both names to the generic implementation in _backends/cupy.py.


def awkward_reduce_prod(
    result,
    input_data,
    offsets_data,
    outlength,
):
    d_input = input_data.astype(result.dtype, copy=False)

    start_o, end_o = make_segment_views(offsets_data)
    h_init = np.asarray(1, dtype=result.dtype)

    def prod_op(a, b):
        return a * b

    segmented_reduce(
        d_in=d_input,
        d_out=result,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=prod_op,
        h_init=h_init,
    )


def awkward_reduce_prod_complex(
    result,
    input_data,
    offsets_data,
    outlength,
):
    complex_dtype = infer_complex_dtype(input_data.dtype)

    input_complex = input_data.view(complex_dtype)
    result_complex = result.view(complex_dtype)
    start_o, end_o = make_segment_views(offsets_data)

    h_init = np.asarray(1.0 + 0.0j, dtype=complex_dtype)

    def prod_op(a, b):
        return a * b

    segmented_reduce(
        d_in=input_complex,
        d_out=result_complex,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=prod_op,
        h_init=h_init,
    )


def awkward_reduce_prod_bool(
    result,
    input_data,
    offsets_data,
    outlength,
):
    # ak.all semantics: result is True for a bin iff every element is non-zero.
    # We must NOT use integer multiplication here — for an int64 input array,
    # the running product overflows mod 2^64 and frequently collapses to 0
    # even when every element is non-zero, giving a spurious False.
    # Instead, map each element to {0, 1} once and reduce with MIN (= AND).
    # This mirrors awkward_reduce_sum_bool's MAX-over-{0,1} for ak.any.
    mapped = cp.empty(input_data.shape, dtype=cp.int8)

    def is_nonzero(x):
        return cp.int8(1) if x != 0 else cp.int8(0)

    unary_transform(
        d_in=input_data,
        d_out=mapped,
        op=is_nonzero,
        num_items=input_data.size,
    )

    d_out = result.view(cp.int8) if result.dtype == cp.bool_ else result
    start_o, end_o = make_segment_views(offsets_data)
    h_init = np.asarray(1, dtype=cp.int8)  # identity for MIN over {0, 1}

    segmented_reduce(
        d_in=mapped,
        d_out=d_out,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=OpKind.MINIMUM,
        h_init=h_init,
    )


def awkward_reduce_prod_bool_complex(
    result,
    input_data,
    offsets_data,
    outlength,
):
    complex_dtype = infer_complex_dtype(input_data.dtype)
    input_complex = input_data.view(complex_dtype)

    mapped_input = cp.empty(input_complex.shape, dtype=cp.int8)

    def is_nonzero(c):
        return cp.int8(c.real != 0 or c.imag != 0)

    unary_transform(
        d_in=input_complex,
        d_out=mapped_input,
        op=is_nonzero,
        num_items=input_complex.size,
    )

    start_o, end_o = make_segment_views(offsets_data)
    d_result = result.view(cp.int8)

    h_init = np.asarray(1, dtype=cp.int8)

    def prod_op(a, b):
        return a * b

    segmented_reduce(
        d_in=mapped_input,
        d_out=d_result,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=prod_op,
        h_init=h_init,
    )


def awkward_reduce_max(
    result,
    input_data,
    offsets_data,
    outlength,
    identity,
):
    start_o, end_o = make_segment_views(offsets_data)

    h_init = np.asarray(identity, dtype=input_data.dtype)

    # def max_op(a, b):
    #     return a if a > b else b

    segmented_reduce(
        d_in=input_data,
        d_out=result,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=OpKind.MAXIMUM,
        h_init=h_init,
    )


def awkward_reduce_max_complex(
    result,
    input_data,
    offsets_data,
    outlength,
    identity,
):
    complex_dtype = infer_complex_dtype(input_data.dtype)
    input_complex = input_data.view(complex_dtype)
    result_complex = result.view(complex_dtype)

    start_o, end_o = make_segment_views(offsets_data)

    h_init = np.asarray(identity + 0.0j, dtype=complex_dtype)

    def lex_max_op(a, b):
        if a.real > b.real:
            return a
        if a.real == b.real and a.imag > b.imag:
            return a
        return b

    segmented_reduce(
        d_in=input_complex,
        d_out=result_complex,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=lex_max_op,
        h_init=h_init,
    )


def awkward_reduce_min(
    result,
    input_data,
    offsets_data,
    outlength,
    identity,
):
    start_o, end_o = make_segment_views(offsets_data)

    h_init = np.asarray(identity, dtype=input_data.dtype)

    segmented_reduce(
        d_in=input_data,
        d_out=result,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=OpKind.MINIMUM,
        h_init=h_init,
    )


def awkward_reduce_min_complex(
    result,
    input_data,
    offsets_data,
    outlength,
    identity,
):
    complex_dtype = infer_complex_dtype(input_data.dtype)
    input_complex = input_data.view(complex_dtype)
    result_complex = result.view(complex_dtype)

    start_o, end_o = make_segment_views(offsets_data)

    h_init = np.asarray(identity + 0.0j, dtype=complex_dtype)

    def lex_min_op(a, b):
        if a.real < b.real:
            return a
        if a.real == b.real and a.imag < b.imag:
            return a
        return b

    segmented_reduce(
        d_in=input_complex,
        d_out=result_complex,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=lex_min_op,
        h_init=h_init,
    )


def awkward_reduce_count_64(
    result,
    offsets_data,
    _outlength,
):
    result[:] = offsets_data[1:] - offsets_data[:-1]


def awkward_reduce_countnonzero(
    result,
    input_data,
    offsets_data,
    outlength,
):
    # Temporary workaround for bool instability
    if input_data.dtype == cp.bool_:
        input_data = input_data.view(cp.int8)

    mapped_data = cp.empty(input_data.shape, dtype=result.dtype)

    def is_nonzero_map(x):
        return result.dtype.type(1) if x != 0 else result.dtype.type(0)

    unary_transform(
        d_in=input_data,
        d_out=mapped_data,
        op=is_nonzero_map,
        num_items=input_data.size,
    )

    start_o, end_o = make_segment_views(offsets_data)
    h_init = np.asarray(0, dtype=result.dtype)

    segmented_reduce(
        d_in=mapped_data,
        d_out=result,
        num_segments=outlength,
        start_offsets_in=start_o,
        end_offsets_in=end_o,
        op=OpKind.PLUS,
        h_init=h_init,
    )


def awkward_reduce_countnonzero_complex(
    result,
    input_data,
    offsets_data,
    outlength,
):
    # Complex values arrive as a flat float32/float64 array of length 2*N
    # (real/imag interleaved). Re-view into complex dtype for reduction.

    complex_dtype = infer_complex_dtype(input_data.dtype)

    input_complex = input_data.view(complex_dtype)

    index_dtype = normalize_index_dtype(offsets_data.dtype)
    start_o, end_o = make_segment_views(offsets_data)

    def segment_reduce_countnonzero(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]

        count = 0

        for i in range(start_idx, end_idx):
            if input_complex[i] != complex_dtype(0):
                count += 1

        return count


# Overlays a mask onto an index array: masked positions become -1, unmasked positions keep their original index value.
def awkward_IndexedArray_overlay_mask(toindex, mask, fromindex, length):
    def transform(i):
        return -1 if mask[i] else fromindex[i]

    indices = CountingIterator(cp.int64(0))
    unary_transform(indices, toindex, transform, length)


# Skips masked (-1) entries and packs the remaining valid entries into nextcarry and nextparents, tracking where each ended up in outindex.
def awkward_IndexedArray_reduce_next_64(
    nextcarry, nextparents, outindex, index, parents, length
):
    if length == 0:
        return

    # Compute cumulative count of valid (non-negative) indices to determine compact output positions
    # this needs to be done before going through all the indices in parallel later
    scan = cp.cumsum(index >= 0)

    def scatter_and_fill(i):
        if index[i] >= 0:
            # Map valid entry to its compacted position
            k = scan[i] - 1
            nextcarry[k] = index[i]
            nextparents[k] = parents[i]
            return k
        # Masked entries get -1 in outindex
        return -1

    indices = CountingIterator(cp.int64(0))
    unary_transform(indices, outindex, scatter_and_fill, length)


# For each valid (non-negative) entry at position i, records the number of null (negative) entries
# that appeared before it. The k-th valid entry gets nextshifts[k] = count of nulls before position i.
# For example, für index = [0, 1, 2, -1, 3, -1, 4] → nextshifts = [0, 0, 0, 1, 2].
def awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64(nextshifts, index, length):
    if length == 0:
        return

    index_slice = index[:length]

    # cumsum of (index < 0) gives the running null count at each position.
    # this is basically equivalent to calling cuda.compute.inclusive_scan on index_slice < 0
    null_cumsum = cp.cumsum(index_slice < 0)
    _ = cp.empty(length, dtype=cp.int64)

    def scatter(i):
        null_count = null_cumsum[i]
        if index_slice[i] >= 0:
            nextshifts[i - null_count] = null_count  # output slot = i - null_count
        # return a dummy value otherwise
        return cp.int64(0)

    indices = CountingIterator(cp.int64(0))
    unary_transform(indices, _, scatter, length)


# Packs valid entries (where (mask[i] != 0) == validwhen) into tocarry in order.
# Examples:
# mask = [0, 1, 0, 1, 1], validwhen=True  → tocarry = [1, 3, 4]
# mask = [0, 1, 0, 1, 1], validwhen=False → tocarry = [0, 2]
# mask = [0, 1, 0, 1, 1, -1, 1], validwhen=True → tocarry = [1, 3, 4, 5, 6]
def awkward_ByteMaskedArray_getitem_nextcarry(tocarry, mask, length, validwhen):
    if length == 0:
        return

    # valid = ((mask[:length] != 0) == validwhen)
    # valid[i] is 1 when the masked element passes the validwhen condition.

    # get the indices of the valid entries using cp.nonzero
    valid_indices = cp.nonzero((mask[:length] != 0) == validwhen)[0]
    # in case tocarry is not exactly the right size, allocate it in two steps like this
    tocarry[: len(valid_indices)] = valid_indices


# Counts null (invalid) entries: positions where (mask[i] != 0) != validwhen.
# Examples:
# mask = [0, 1, 0, 1, 1], validwhen=True  → numnull = 2  (positions 0 and 2 are null)
# mask = [0, 1, 0, 1, 1], validwhen=False → numnull = 3  (positions 1, 3 and 4 are null)
def awkward_ByteMaskedArray_numnull(numnull, mask, length, validwhen):
    numnull[0] = cp.count_nonzero((mask[:length] != 0) != validwhen)


# Broadcasts a single jagged offset array across all rows of a regular array
# Example:
# singleoffsets = [0, 2, 5], regularsize = 2, regularlength = 3
# multistarts = [0, 2, 0, 2, 0, 2]
# multistops  = [2, 5, 2, 5, 2, 5]
def awkward_RegularArray_getitem_jagged_expand(
    multistarts, multistops, singleoffsets, regularsize, regularlength
):
    if regularlength == 0 or regularsize == 0:
        return

    # Reshape as (regularlength, regularsize) views (no copy) and broadcast-assign
    # singleoffsets[:-1] / singleoffsets[1:] across all rows.
    multistarts.reshape(regularlength, regularsize)[:] = singleoffsets[:regularsize]
    multistops.reshape(regularlength, regularsize)[:] = singleoffsets[
        1 : regularsize + 1
    ]


# THIS KERNEL IS NOT USED (just for archive)
# Fills a tagged index for one union type: assigns a constant tag and
# sequential index into each segment defined by the starts/counts ranges
# Example input:
# tmpstarts = [0, 3], tag = 1, fromcounts = [3, 2]
# Example output:
# totags  = [1, 1, 1, 1, 1]
# toindex = [0, 1, 2, 0, 1]
# also, the tmpstarts get rewritten with stops: tmpstarts = [3, 5]
def awkward_UnionArray_nestedfill_tags_index(
    totags, toindex, tmpstarts, tag, fromcounts, length
):
    if length == 0:
        return

    starts = tmpstarts[:length]
    counts = fromcounts[:length]

    # Total span of the output arrays we need to touch:
    # the last segment's start + its count gives the furthest written position
    total_size = int(starts[length - 1]) + int(counts[length - 1])

    if total_size == 0:
        return

    # +1 at each segment start, -1 just past each segment end.
    # cumsum of this will later yield 1 inside any covered range, 0 in gaps.
    diff = cp.zeros(total_size + 1, dtype=cp.int8)

    def scatter_and_update(i):
        start = starts[i]
        count = counts[i]
        # Mark this segment's range in the difference array
        diff[start] += cp.int8(1)
        diff[start + count] -= cp.int8(1)
        # update tmpstarts (for the next call of this kernel (for a different union type))?
        tmpstarts[i] = start + count
        return 0

    # Scatter segment's ranges and update tmpstarts
    unary_transform(
        CountingIterator(cp.int64(0)), DiscardIterator(), scatter_and_update, length
    )

    # coverage[j] == 1 if position j falls inside any segment's range, 0 otherwise
    coverage = cp.cumsum(diff[:total_size])

    # scan[j] == local index of element j within its segment
    # Since it's a cumsum, the first index starts from 1, 2, 3 ...
    # so we'll have to -1 before writing it in toindex
    scan = cp.cumsum(coverage, dtype=cp.int64)

    def fill(j):
        if coverage[j]:
            # Mark this position as belonging to the current tag
            totags[j] = tag
            toindex[j] = scan[j] - 1
        return 0

    unary_transform(CountingIterator(cp.int64(0)), DiscardIterator(), fill, total_size)


# For each position i where fromtags[i] == fromwhich, sets totags[i] = towhich and
# toindex[i] = fromindex[i] + base. Other positions are left unchanged.
# Example:
# fromtags  = [0, 1, 0, 1, 0], fromindex = [0, 0, 1, 1, 2]
# fromwhich=1, towhich=2, base=10
# totags  = [0, 2, 0, 2, 0]
# toindex = [0, 10, 1, 11, 2]
def awkward_UnionArray_simplify_one(
    totags, toindex, fromtags, fromindex, towhich, fromwhich, length, base
):
    if length == 0:
        return

    def transform(i):
        if fromtags[i] == fromwhich:
            totags[i] = towhich
            toindex[i] = fromindex[i] + base
        return 0  # discarded

    indices = CountingIterator(cp.int64(0))
    unary_transform(indices, DiscardIterator(), transform, length)


# producing a carry index that maps each output element back to its position in the original content
# Example input:
# fromoffsets = [0, 3, 5], fromstarts = [10, 20], fromstops = [13, 22], lencontent = 25
# Example output:
# i=0: range [10, 13) → [10, 11, 12]
# i=1: range [20, 22) → [20, 21]
# tocarry = [10, 11, 12, 20, 21]
def awkward_ListArray_broadcast_tooffsets(
    tocarry, fromoffsets, offsetslength, fromstarts, fromstops, lencontent
):
    if offsetslength <= 1:
        return

    length = offsetslength - 1
    starts = fromstarts[:length]
    stops = fromstops[:length]
    # counts[i] = how many elements list i should have
    counts = fromoffsets[1:offsetslength] - fromoffsets[:length]

    _K = "awkward_ListArray_broadcast_tooffsets"
    if cp.any((starts != stops) & (stops > lencontent)):
        raise ValueError(f"stops[i] > len(content) in compiled CUDA code ({_K})")
    if cp.any(counts < 0):
        raise ValueError(
            f"broadcast's offsets must be monotonically increasing in compiled CUDA code ({_K})"
        )
    if cp.any(stops - starts != counts):
        raise ValueError(f"cannot broadcast nested list in compiled CUDA code ({_K})")

    # For each segment i, write the content indices starts[i], starts[i]+1, ..., stops[i]-1
    # into the contiguous output slice tocarry[fromoffsets[i] : fromoffsets[i+1]].
    def fill_list(i):
        start = starts[i]
        stop = stops[i]
        for j in range(start, stop):
            tocarry[fromoffsets[i] + j - start] = j
        return 0

    unary_transform(CountingIterator(cp.int64(0)), DiscardIterator(), fill_list, length)


# For each segment i, it fills toindex with the local position of each element within that segment — i.e. 0, 1, 2, ...
# Example:
# offsets = [0, 3, 5]
# toindex = [0, 1, 2, 0, 1]
def awkward_ListArray_localindex(toindex, offsets, length):
    if length == 0:
        return

    starts = offsets[:length]
    stops = offsets[1 : length + 1]

    def fill(i):
        start = starts[i]
        stop = stops[i]
        for j in range(start, stop):
            toindex[j] = j - start
        return 0

    unary_transform(CountingIterator(cp.int64(0)), DiscardIterator(), fill, length)


# Converts a ListArray's (starts, stops) pairs into offsets.
# tooffsets[0] = 0, tooffsets[i+1] = tooffsets[i] + (fromstops[i] - fromstarts[i])
# Example:
# fromstarts = [10, 20], fromstops = [13, 22], length = 2
# tooffsets = [0, 3, 5]
def awkward_ListArray_compact_offsets(tooffsets, fromstarts, fromstops, length):
    tooffsets[0] = 0
    if length == 0:
        return

    starts = fromstarts[:length]
    stops = fromstops[:length]

    if cp.any(stops < starts):
        raise ValueError(
            "stops[i] < starts[i] in compiled CUDA code (awkward_ListArray_compact_offsets)"
        )

    sizes = stops - starts

    # the same as `tooffsets[1 : length + 1] = cp.cumsum(sizes)`
    inclusive_scan(
        sizes,
        tooffsets[1 : length + 1],
        lambda a, b: a + b,
        cp.array([0], dtype=tooffsets.dtype),
        length,
    )


# For each list i, counts the number of n-combinations of its elements
# (with or without replacement) and builds an offsets array into tooffsets.
# totallen[0] is set to the total number of combinations across all lists.
#
# Example (n=2, replacement=False):
# starts=[0, 0, 0], stops=[2, 3, 4]
# sizes = [2, 3, 4]
# C(2,2)=1, C(3,2)=3, C(4,2)=6
# Then the output will be: tooffsets = [0, 1, 4, 10]
# totallen  = 10
def awkward_ListArray_combinations_length(
    totallen, tooffsets, n, replacement, starts, stops, length
):
    tooffsets[0] = 0
    if length == 0:
        totallen[0] = 0
        return

    def combinations_len(i):
        size = stops[i] - starts[i]
        if replacement:
            size = size + (n - 1)
        thisn = n
        if thisn > size:
            return 0
        elif thisn == size:
            return 1
        else:
            # C(size, n) == C(size, size-n), so use the smaller one
            # of the two to minimise the number of loop iterations
            if thisn * 2 > size:
                thisn = size - thisn

            # Compute C(size, thisn) = size! / (thisn! * (size-thisn)!) incrementally:
            # result = size * (size-1) * ... * (size-thisn+1) / thisn!
            result = size
            for j in range(2, thisn + 1):
                result = result * (size - j + 1)
                result = result // j
            return result

    # Compute the number of combinations for each list
    counts = cp.empty(length, dtype=tooffsets.dtype)
    unary_transform(CountingIterator(cp.int64(0)), counts, combinations_len, length)

    # Convert counts to offsets:
    # tooffsets[i+1] = sum(counts[0..i])
    inclusive_scan(
        counts,
        tooffsets[1 : length + 1],
        lambda a, b: a + b,
        cp.array([0], dtype=tooffsets.dtype),
        length,
    )

    # Total number of combinations across all lists
    totallen[0] = tooffsets[length]


# For each list i, enumerates all n-combinations (with or without replacement)
# of its elements and writes the indices into n output carry arrays.
#
# tocarry_ptrs is a CuPy int64 array of length n holding raw device pointers;
# each pointer refers to a pre-allocated int64 array of length totallen.
#
# Example (n=2, replacement=False):
# starts=[0], stops=[3]  → elements [0,1,2]
# C(3,2) = 3 combinations in total
# combinations: (0,1),(0,2),(1,2)
#
# Output:
# tocarry_ptrs[0] → [0, 0, 1],  tocarry_ptrs[1] → [1, 2, 2]
# toindex: [3, 3]
def awkward_ListArray_combinations(
    tocarry_ptrs, toindex, fromindex, n, replacement, starts, stops, length
):
    if length == 0:
        return

    # Step 1: compute per-list combination counts (same as combinations_length!!)
    # TODO: we can just pass combination offsets directly in the future (from src/awkward/contents/listoffsetarray.py:1405)
    def combinations_len(i):
        size = stops[i] - starts[i]
        if replacement:
            size = size + (n - 1)
        thisn = n
        if thisn > size:
            return 0
        elif thisn == size:
            return 1
        else:
            if thisn * 2 > size:
                thisn = size - thisn
            result = size
            for j in range(2, thisn + 1):
                result = result * (size - j + 1)
                result = result // j
            return result

    counts = cp.empty(length, dtype=cp.int64)
    unary_transform(CountingIterator(cp.int64(0)), counts, combinations_len, length)

    offsets = cp.empty(length + 1, dtype=cp.int64)
    offsets[0] = 0
    inclusive_scan(
        counts,
        offsets[1:],
        lambda a, b: a + b,
        cp.array([0], dtype=cp.int64),
        length,
    )

    totallen = int(offsets[length])
    if totallen == 0:
        return

    # Step 2: wrap raw pointers from tocarry_ptrs into CuPy arrays
    # raw int64 pointer values from tocarry_ptrs[k] can't be dereferenced inside a Numba closure, so
    # we need this intermediate step
    #
    # (the pointers themselves are allocated at src/awkward/contents/listoffsetarray.py:1456-1464)
    carry_arrays = []
    for k in range(n):
        ptr_val = int(tocarry_ptrs[k])
        mem = cp.cuda.UnownedMemory(ptr_val, totallen * 8, None)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        carry_arrays.append(cp.ndarray(totallen, dtype=cp.int64, memptr=memptr))  # pylint: disable=unexpected-keyword-arg

    # -------------------------------------------------------------------------
    # Step 3: fill carry_arrays[k] for each combination position k in turn.
    #
    # For each output slot g in [0, totallen):
    #
    #   a) Binary search offsets to find which source list i owns slot g,
    #      and compute the rank of this combination within that list
    #      (rank = g - offsets[i], i.e. the 0-based index among all combinations
    #      of list i in lexicographic order).
    #
    #   b) Unrank: decode the rank back into the actual combination tuple using
    #      a combinatorial number system. Iterating over positions pos=0..n-1,
    #      at each position scan forward through candidate values j, counting
    #      how many combinations start with values < j at this position
    #      (= C(effective_size-j-1, n-pos-1)). Subtract from remaining rank
    #      until we find the j where remaining < count — that j is the value
    #      at position pos.
    #
    #   c) Early exit: once pos==k we have the value for position k and write
    #      it to carry_k[g], skipping the rest of the unranking. This is why
    #      we do n separate passes (one per k) rather than one pass writing all
    #      n positions: each pass only needs to unrank up to position k.
    #
    #   d) Content index: add start (the list's base offset into content) to
    #      convert from a within-list index to an absolute content index.
    #      For replacement, subtract pos to undo the stars-and-bars shift.
    # -------------------------------------------------------------------------
    def make_pass(k, carry_k):
        def fill_pos(g):
            # a) Find source list i via binary search on offsets
            lo = 0
            hi = length - 1
            while lo < hi:
                mid = (lo + hi) >> 1
                if offsets[mid + 1] <= g:
                    lo = mid + 1
                else:
                    hi = mid
            list_i = lo
            start = starts[list_i]
            size = stops[list_i] - starts[list_i]
            rank = g - offsets[list_i]
            # For replacement use stars-and-bars effective size
            effective_size = size + n - 1 if replacement else size

            # b) Unrank: decode rank into the combination tuple
            lower = 0  # lower bound for j at each position (enforces ordering)
            remaining = rank
            for pos in range(n):
                for j in range(lower, effective_size - (n - pos - 1)):
                    # Count combinations where position pos has value j:
                    # = C(effective_size - j - 1, n - pos - 1)
                    top = effective_size - j - 1
                    choose = n - pos - 1
                    if choose == 0:
                        count = 1
                    else:
                        if choose * 2 > top:  # use smaller equivalent
                            choose = top - choose
                        c = top
                        for q in range(2, choose + 1):
                            c = c * (top - q + 1)
                            c = c // q
                        count = c
                    if remaining < count:
                        # c) j is the value at position pos
                        if pos == k:
                            # d) write absolute content index and exit early
                            carry_k[g] = (j - pos if replacement else j) + start
                            return 0
                        lower = j + 1  # next position must be >= j+1 (no repeat)
                        break
                    remaining -= count
            return 0

        return fill_pos

    # One parallel pass per combination position k
    for k in range(n):
        unary_transform(
            CountingIterator(cp.int64(0)),
            DiscardIterator(),
            make_pass(k, carry_arrays[k]),
            totallen,
        )

    toindex[:n] = totallen
