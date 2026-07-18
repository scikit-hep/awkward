# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""NumPy implementations of the segment algorithms used by the lazy IR.

These are the CPU counterparts of ``awkward._connect.cuda.helpers`` /
``_segment_algorithms``.  All functions are vectorized (no per-list Python
loops) and handle empty lists.
"""

from __future__ import annotations

import numpy as np

import awkward as ak

# Layout utilities are backend-neutral and shared with the CUDA backend.
from awkward._connect.lazy._layout import (  # noqa: F401
    empty_like,
    reconstruct_with_offsets,
)


def _listoffset_parts(array):
    """Extract the flat parts of a list-of-numbers array.

    Args:
        array (ak.Array or ak.contents.Content): A list-type array.

    Returns a ``(layout, offsets, content)`` tuple: the normalized
    ``ListOffsetArray64`` layout, its offsets, and the flattened NumPy content.

    Raises:
        TypeError: If ``array`` is not a list-type array.
        NotImplementedError: If the list content is not a ``NumpyArray``.
    """
    if isinstance(array, ak.Array):
        layout = array.layout
    elif isinstance(array, ak.contents.Content):
        layout = array
    else:
        layout = ak.to_layout(array)

    if not isinstance(
        layout,
        (ak.contents.ListOffsetArray, ak.contents.ListArray, ak.contents.RegularArray),
    ):
        raise TypeError(f"expected a list-type array, got {type(layout).__name__}")

    layout = layout.to_ListOffsetArray64(True)

    if not isinstance(layout.content, ak.contents.NumpyArray):
        raise NotImplementedError(
            f"list content of type {type(layout.content).__name__} "
            f"is not yet supported (expected NumpyArray)"
        )

    offsets = np.asarray(layout.offsets.data)
    content = np.asarray(layout.content.data)[: offsets[-1]]
    return layout, offsets, content


def _rebuild(layout, offsets, content):
    return ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(offsets.astype(np.int64, copy=False)),
            ak.contents.NumpyArray(content),
            parameters=layout._parameters,
        )
    )


def segment_sizes(offsets):
    """
    Args:
        offsets: Segment offsets (length = num_segments + 1).

    Returns the size of each segment (length = num_segments).
    """
    return offsets[1:] - offsets[:-1]


def list_sizes(array):
    """
    Args:
        array (ak.Array): A list array.

    Returns a NumPy array of per-list element counts.
    """
    _, offsets, _ = _listoffset_parts(array)
    return segment_sizes(offsets)


def filter_lists(array, cond):
    """Keep elements within each list for which ``cond`` is true.

    Args:
        array (ak.Array): A list array.
        cond (callable): Vectorized predicate over a NumPy array of the flat
            content (e.g. ``lambda x: x > 2``), mirroring the CUDA predicate
            semantics.

    Returns a new ``ak.Array`` with the kept elements and updated offsets.
    """
    layout, offsets, content = _listoffset_parts(array)

    mask = np.asarray(cond(content), dtype=bool)

    # kept[i] = number of kept elements among the first i elements, so
    # gathering it at the old offsets yields the new offsets — empty lists
    # (repeated offsets) fall out naturally.
    kept = np.zeros(len(content) + 1, dtype=np.int64)
    np.cumsum(mask, out=kept[1:])

    return _rebuild(layout, kept[offsets], content[mask])


def select_lists(array, mask):
    """Keep entire lists selected by a per-list mask.

    Args:
        array (ak.Array): A list array.
        mask: A per-list boolean mask; its length must equal the number of
            lists.

    Returns a new ``ak.Array`` containing only the selected lists.

    Raises:
        ValueError: If ``mask`` length does not match the number of lists.
    """
    layout, offsets, content = _listoffset_parts(array)

    mask = np.asarray(mask, dtype=bool)
    num_lists = len(offsets) - 1
    if len(mask) != num_lists:
        raise ValueError(
            f"mask length {len(mask)} does not match number of lists {num_lists}"
        )

    sizes = segment_sizes(offsets).astype(np.intp, copy=False)
    element_mask = np.repeat(mask, sizes)

    kept_sizes = sizes[mask]
    new_offsets = np.zeros(len(kept_sizes) + 1, dtype=np.int64)
    np.cumsum(kept_sizes, out=new_offsets[1:])

    return _rebuild(layout, new_offsets, content[element_mask])


def transform_lists(array, out, list_size, op):
    """Apply an n-ary ``op`` across the items of equal-size lists.

    Args:
        array (ak.Array): A list array; every list must contain exactly
            ``list_size`` items.
        out (numpy.ndarray): Writeable output buffer of length num_lists; the
            result is written in place.
        list_size (int): The common list length.
        op (callable): Receives one argument per item position (each a NumPy
            array over all lists) and returns one value per list, e.g.
            ``lambda x, y, z: x + y + z``.

    Returns ``out`` with the per-list results.

    Raises:
        ValueError: If any list does not have exactly ``list_size`` items.
        TypeError: If ``out`` is not a writeable NumPy array.
    """
    _, offsets, content = _listoffset_parts(array)
    num_segments = len(offsets) - 1

    # The equal-size precondition is load-bearing: a ragged reshape would
    # silently mix elements across lists.  Check it rather than corrupt.
    sizes = segment_sizes(offsets)
    if not np.all(sizes == list_size):
        raise ValueError(
            f"transform_lists requires every list to have exactly {list_size} "
            "items; got ragged input"
        )

    columns = content[: num_segments * list_size].reshape(num_segments, list_size)
    result = op(*(columns[:, i] for i in range(list_size)))

    # ``out`` must be writeable in place; a non-ndarray would be copied by
    # np.asarray and the write lost, so require an ndarray.
    if not isinstance(out, np.ndarray):
        raise TypeError("transform_lists `out` must be a writeable NumPy array")
    out[:num_segments] = result
    return out
