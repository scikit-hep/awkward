# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import cupy as cp
import numpy as np

from cuda.compute import (
    DiscardIterator, PermutationIterator, ZipIterator, TransformIterator, CountingIterator,
    OpKind, exclusive_scan, segmented_reduce, select, unary_transform
)


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


def offsets_to_segment_ids(offsets, stream=None):
    """
    Convert segment offsets to segment IDs (indicators).

    Given offsets [0, 2, 5, 8, 10], produces [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]

    This function correctly handles empty segments (duplicate offsets).
    For example, offsets [0, 0, 2, 3] with an empty first segment produces [0, 0, 1]
    where elements at indices 0-1 belong to segment 1 (the empty segment 0 has no elements).

    The implementation uses binary search to find which segment each element belongs to.
    For each element at index i, we find segment j where offsets[j] <= i < offsets[j+1].

    Args:
        offsets: Device array of segment offsets (length = num_segments + 1).
                 The last element is the total number of elements.
        stream: CUDA stream for the operation (optional).

    Returns:
        Device array of segment IDs for each element (length = offsets[-1]).
    """
    num_elements = int(offsets[-1])

    if num_elements == 0:
        return cp.array([], dtype=np.int32)

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
    [[30, 20], [20, 50, 90], [10, 30, 80], [20, 60]]

    This function selects entire segments based on a mask, keeping only
    the segments where mask[i] is non-zero. The results are written to the provided
    output arrays/iterators.

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

    # Step 3: Filter the data array using the expanded mask
    # We use a ZipIterator to combine data and mask, then filter based on mask
    # and extract only the data values

    # Zip data and expanded mask together
    zip_it = ZipIterator(data_in, expanded_mask_it)

    # Allocate temporary mask output buffer
    filtered_pairs = ZipIterator(data_out, DiscardIterator())
    d_num_data_selected = cp.zeros(1, dtype=np.int32)

    # Define predicate that checks if mask value is non-zero
    def mask_predicate(pair):
        return pair[1] != 0

    # Apply select to the zipped iterator
    select(
        zip_it,
        filtered_pairs,
        d_num_data_selected,
        mask_predicate,
        num_elements,
        stream,
    )

    # Get the actual number of selected elements
    num_selected = int(d_num_data_selected[0])

    # Step 4: Compute new segment offsets
    # For each original segment, count how many of its elements passed the selection
    # This is a segmented reduction of the expanded_mask with SUM operation

    # Segmented reduce to count elements per segment that passed the selection
    segment_counts = cp.zeros(num_segments, dtype=np.int32)
    h_init = np.array([0], dtype=np.int32)

    # For segmented_reduce, we need start and end offsets
    # If offsets_in is an iterator, we can still slice it
    start_offsets = offsets_in[:-1]
    end_offsets = offsets_in[1:]

    segmented_reduce(
        expanded_mask_it,
        segment_counts,
        start_offsets,
        end_offsets,
        OpKind.PLUS,
        h_init,
        num_segments,
        stream,
    )

    # Select out the segment sizes where mask is zero (those segments are not included)
    kept_segment_sizes = cp.empty(num_segments, dtype=np.int32)
    d_num_kept_segments = cp.zeros(1, dtype=np.int32)

    # Use ZipIterator to combine segment_counts and mask for selection
    zip_sizes_mask = ZipIterator(segment_counts, mask_in)

    def mask_nonzero(pair):
        return pair[1] != 0

    select(
        zip_sizes_mask,
        ZipIterator(kept_segment_sizes, cp.empty(num_segments, dtype=np.int8)),
        d_num_kept_segments,
        mask_nonzero,
        num_segments,
        stream,
    )

    num_kept_segments = int(d_num_kept_segments[0])
    kept_segment_sizes = kept_segment_sizes[:num_kept_segments]

    # Exclusive scan to convert sizes to offsets
    temp_offsets = cp.zeros(num_kept_segments + 1, dtype=np.int32)

    if num_kept_segments > 0:
        h_init_scan = np.array([0], dtype=np.int32)
        exclusive_scan(
            kept_segment_sizes,
            temp_offsets[:-1],
            OpKind.PLUS,
            h_init_scan,
            num_kept_segments,
            stream,
        )
        # Set the final offset to the total number of selected elements
        temp_offsets[-1] = num_selected

    # Copy results to output iterators/arrays
    # For offsets_out, we need to copy the computed offsets
    offsets_out[: num_kept_segments + 1] = temp_offsets

    # Store the counts in d_num_selected_out
    d_num_selected_out[0] = num_selected  # number of data elements
    d_num_selected_out[1] = num_kept_segments  # number of segments kept


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
    num_segments = len(d_in_segments) - 1

    # Step 1: Apply select to compact the data
    d_num_selected = cp.zeros(2, dtype=cp.uint64)
    select(d_in_data, d_out_data, d_num_selected, cond, num_items, stream)

    # Get total number of selected items
    total_selected = int(d_num_selected[0])

    # Step 2: Create a boolean mask by applying the condition
    # We materialize this as an array to avoid closure issues
    d_mask = cp.empty(num_items, dtype=cp.uint8)
    unary_transform(d_in_data, d_mask, cond, num_items, stream)

    # Step 3: Use segmented reduce to count selected items per segment
    start_offsets = d_in_segments[:-1]
    end_offsets = d_in_segments[1:]
    d_counts = cp.empty(num_segments, dtype=cp.uint64)

    segmented_reduce(
        d_mask,
        d_counts,
        start_offsets,
        end_offsets,
        OpKind.PLUS,
        np.array(0, dtype=np.uint64),
        num_segments,
        stream,
    )

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


def transform_segments(data_in, data_out, segment_size, op, num_segments):
    def column_it_factory(it, i):
        def col_it(j: np.int32) -> np.int32:
            return j * segment_size + i
        return PermutationIterator(it, TransformIterator(CountingIterator(np.int32(0)), col_it))

    column_iterators = []
    for i in range(segment_size):
        column_iterators.append(column_it_factory(data_in, i))

    columns = ZipIterator(*column_iterators)
    unary_transform(
        columns,
        data_out,
        op,
        num_segments
    )
