# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import cupy as cp
import numpy as np
import nvtx

from cuda.compute import (
    PermutationIterator, ZipIterator, TransformIterator, CountingIterator,
    OpKind, exclusive_scan, select, unary_transform
)


@nvtx.annotate("segment_sizes")
def segment_sizes(offsets):
    """
    Compute the size of each segment from segment offsets.

    Args:
        offsets: Device array of segment offsets (length = num_segments + 1).
                 Each segment i contains elements from offsets[i] to offsets[i+1].

    Returns:
        Device array of segment sizes (length = num_segments).
    """
    return offsets[1:] - offsets[:-1]


@nvtx.annotate("offsets_to_segment_ids")
def offsets_to_segment_ids(offsets, stream=None):
    """
    Convert segment offsets to segment IDs (indicators).

    Given offsets [0, 2, 5, 8, 10], produces [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    """
    num_elements = int(offsets[-1])

    if num_elements == 0:
        return cp.array([], dtype=np.int32)

    # TODO: when available this can be a fused CountingIterator + lower_bound

    # Create array of all element indices [0, 1, 2, ..., num_elements-1]
    element_indices = cp.arange(num_elements, dtype=np.int32)

    # Use binary search to find which segment each element belongs to
    # searchsorted finds the rightmost position where we can insert each element index
    # such that the array remains sorted. Subtracting 1 gives us the segment ID.
    #
    # Example: offsets = [0, 0, 2, 3], element_indices = [0, 1, 2]
    # searchsorted(offsets[:-1], [0, 1, 2], side='right') = [1, 1, 2]
    # Subtracting 1: [0, 0, 1] - but wait, element 0 should be in segment 1!
    #
    # Actually, we want: offsets[j] <= i < offsets[j+1]
    # So searchsorted with side='right' on offsets (not offsets[:-1]) and subtract 1
    segment_ids = cp.searchsorted(offsets, element_indices, side="right") - 1

    return segment_ids


@nvtx.annotate("select_segments")
def select_segments(
    data_in,
    offsets_in,
    mask_in,
    data_out,
    offsets_out,
    d_num_selected_out,
    num_elements,
    num_segments,
    stream=None,
):
    """
    Select segments of a segmented array using a per-segment mask.

    A segmented array is conceptually composed of a data array and segment offsets.
    For example, with data=[30, 20, 20, 50, 90, 10, 30, 80, 20, 60] and
    offsets=[0, 2, 5, 8, 10], the segmented array represents:
    [[30, 20], [20, 50, 90], [10, 30, 80], [20, 60]].
    Given the mask [0, 1, 0, 1], the function will return the segmented array
    [[20, 50, 90], [20, 60]].

    Example:
        >>> data_in = cp.array([30, 20, 20, 50, 90, 10, 30, 80, 20, 60], dtype=np.int32)
        >>> offsets_in = cp.array([0, 2, 5, 8, 10], dtype=np.int32)
        >>> mask_in = cp.array([0, 1, 0, 1], dtype=np.int8)
        >>> data_out = cp.empty_like(data_in)
        >>> offsets_out = cp.empty_like(offsets_in)
        >>> d_num_selected = cp.zeros(2, dtype=np.int32)
        >>> select_segments(data_in, offsets_in, mask_in, data_out, offsets_out,
        ...                 d_num_selected, len(data_in), len(offsets_in) - 1)
        >>> # Result: data_out contains [20, 50, 90, 20, 60, ...]
        >>> #         offsets_out contains [0, 3, 5, ...]
        >>> #         d_num_selected[0] = 5 (number of data elements)
        >>> #         d_num_selected[1] = 2 (number of segments kept)

    Args:
        data_in: Device array or iterator containing all segment elements concatenated.
        offsets_in: Device array or iterator of segment offsets (length = num_segments + 1).
                    Each segment i contains elements from offsets[i] to offsets[i+1].
        mask_in: Device array or iterator (int8) indicating which segments to keep (length = num_segments).
                 Non-zero values indicate segments to keep.
        data_out: Device array or iterator to store selected data elements.
                  Should be pre-allocated with size >= num_elements.
        offsets_out: Device array or iterator to store new segment offsets.
                     Should be pre-allocated with size >= num_segments + 1.
        d_num_selected_out: Device array to store counts (length >= 2):
                           - d_num_selected_out[0]: number of selected data elements
                           - d_num_selected_out[1]: number of segments kept
        num_elements: Total number of elements in data_in.
        num_segments: Total number of segments (= len(offsets_in) - 1).
        stream: CUDA stream for the operation (optional).
    """
    # Step 1: Create segment_indices array indicating which segment each element belongs to
    segment_indices = offsets_to_segment_ids(offsets_in, stream)

    # Step 2: Expand mask from per-segment to per-element using PermutationIterator
    # Each element gets the mask value of its corresponding segment
    expanded_mask_it = PermutationIterator(mask_in, segment_indices)

    # Step 3: Filter the data array and capture indices in a single select call
    # Zip together data, expanded mask, and counting iterator
    data_mask_idx_in = ZipIterator(
        data_in, expanded_mask_it, CountingIterator(np.int32(0)))
    d_selected_indices = cp.empty(num_elements, dtype=np.int32)
    data_idx_out = ZipIterator(data_out, d_selected_indices)
    d_num_data_selected = cp.zeros(1, dtype=np.int32)

    # Define predicate that checks if mask value is non-zero
    def mask_predicate(triple):
        return triple[1] != 0

    # Apply select to get both data and indices where mask is non-zero
    select(
        data_mask_idx_in,
        data_idx_out,
        d_num_data_selected,
        mask_predicate,
        num_elements,
        stream,
    )

    # Get the actual number of selected elements
    num_selected = int(d_num_data_selected[0])
    d_selected_indices = d_selected_indices[:num_selected]

    # Step 4: Compute new segment offsets using the captured indices
    # TODO: this part should use run_length_encode when available

    # Use searchsorted to count elements per segment
    # Use side='left' to count elements strictly less than each offset boundary
    positions = cp.searchsorted(d_selected_indices, offsets_in, side='left')
    segment_counts = (positions[1:] - positions[:-1]).astype(np.int32)

    # Select out the segment sizes where mask is non-zero (those segments are kept)
    # Convert mask to a regular array if needed and use boolean indexing
    mask_array = cp.asarray(mask_in)
    kept_segment_sizes = segment_counts[mask_array != 0]
    num_kept_segments = len(kept_segment_sizes)

    # Exclusive scan to convert sizes to offsets
    temp_offsets = cp.zeros(num_kept_segments + 1, dtype=np.int32)

    if num_kept_segments > 0:
        h_init_scan = np.array([0], dtype=np.int32)
        exclusive_scan(
            kept_segment_sizes,
            offsets_out,
            OpKind.PLUS,
            h_init_scan,
            num_kept_segments,
            stream,
        )

    # Set the final offset to the total number of selected elements
    offsets_out[num_kept_segments] = num_selected

    # Store the counts in d_num_selected_out
    d_num_selected_out[0] = num_selected  # number of data elements
    d_num_selected_out[1] = num_kept_segments  # number of segments kept


@nvtx.annotate("segmented_select")
def segmented_select(
    d_in_data,
    d_in_segments,
    d_out_data,
    d_out_segments,
    cond,
    num_items: int,
    stream=None,
) -> int:
    """
    Select data within segments independently based on a condition.

    Given segmented input data and a selection condition, this function
    applies the selection to each segment independently and produces compacted
    output with updated segment offsets.

    Args:
        d_in_data: Device array containing the input data items.
        d_in_segments: Device array of segment offsets. For N segments,
            this array has N+1 elements where segments[i:i+1] defines
            the range [start, end) for segment i.
        d_out_data: Device array to store selected data (pre-allocated,
            should be at least as large as d_in_data).
        d_out_segments: Device array to store output segment offsets
            (pre-allocated, same size as d_in_segments).
        cond: Callable that takes a data item and returns a boolean-like
            value (typically uint8) indicating whether to keep the item.
        num_items: Total number of items in d_in_data.
        stream: CUDA stream for the operation (optional).

    Returns:
        int: Total number of items after selection (equal to d_out_segments[-1]).

    Example:
        >>> # Input: [[45], [25, 35], [15]] with condition x > 30
        >>> # Output: [[45], [35], []] -> offsets [0, 1, 2, 2]
        >>> def greater_than_30(x):
        ...     return x > 30
        >>> d_in_data = cp.array([45, 25, 35, 15], dtype=cp.int32)
        >>> d_in_segments = cp.array([0, 1, 3, 4], dtype=cp.int32)
        >>> d_out_data = cp.empty_like(d_in_data)
        >>> d_out_segments = cp.empty_like(d_in_segments)
        >>> total = segmented_select(
        ...     d_in_data, d_in_segments, d_out_data, d_out_segments,
        ...     greater_than_30, len(d_in_data)
        ... )
        >>> print(total)  # 2
        >>> print(d_out_segments.get())  # [0, 1, 2, 2]
    """
    import numba.cuda

    num_segments = len(d_in_segments) - 1

    cond = numba.cuda.jit(cond)
    # Apply select to get the data and indices where condition is true

    def select_predicate(pair):
        return cond(pair[0])

    data_idx_in = ZipIterator(d_in_data, CountingIterator(np.int32(0)))
    d_indices_out = cp.empty(num_items, dtype=np.int32)
    data_idx_out = ZipIterator(d_out_data, d_indices_out)
    d_num_selected = cp.zeros(1, dtype=cp.uint64)
    select(data_idx_in, data_idx_out,
           d_num_selected, select_predicate, num_items, stream)

    total_selected = int(d_num_selected[0])
    d_indices_out = d_indices_out[:total_selected]
    d_selected_indices = d_indices_out[:total_selected]

    # Step 3: Use searchsorted to count selected items per segment
    # Use side='left' to count elements strictly less than each offset boundary
    positions = cp.searchsorted(
        d_selected_indices, d_in_segments, side='left')
    d_counts = (positions[1:] - positions[:-1]).astype(cp.uint64)

    # Step 4: Use exclusive scan to compute output segment start offsets
    exclusive_scan(
        d_counts,
        d_out_segments[:-1],
        OpKind.PLUS,
        np.array(0, dtype=np.uint64),
        num_segments,
        stream,
    )

    # Step 5: Set the final offset to the total count
    d_out_segments[-1] = total_selected
    return total_selected


@nvtx.annotate("transform_segments")
def transform_segments(data_in, data_out, segment_size, op, num_segments):
    """
    Given a segmented array where each segment contains the same number of items,
    transform each segment independently using the given n-ary operation.

    For example, given the segmented array [[1, 2, 3], [4, 5, 6], [7, 8, 9]] and the
    operation x + y + z, the function will return the segmented array [[10], [15], [24]].
    """

    def get_column(it, i):
        # return an iterator representing the i-th column of the segmented array.
        def col_major_index(j: np.int32) -> np.int32:
            # given the row major index j, return the column major index.
            return j * segment_size + i
        return PermutationIterator(it, TransformIterator(CountingIterator(np.int32(0)), col_major_index))

    columns = ZipIterator(
        *[get_column(data_in, i) for i in range(segment_size)]
    )
    return unary_transform(columns, data_out, op, num_segments)
