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
    """Return (layout, offsets, flat content) for a list-of-numbers array."""
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
    """Size of each segment from segment offsets (length = num_segments)."""
    return offsets[1:] - offsets[:-1]


def list_sizes(array):
    _, offsets, _ = _listoffset_parts(array)
    return segment_sizes(offsets)


def filter_lists(array, cond):
    """Keep elements within each list for which `cond` is true.

    `cond` must be vectorized over a NumPy array of the flat content
    (e.g. ``lambda x: x > 2``), mirroring the CUDA predicate semantics.
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
    """Keep entire lists selected by the per-list `mask`."""
    layout, offsets, content = _listoffset_parts(array)

    mask = np.asarray(mask, dtype=bool)
    num_lists = len(offsets) - 1
    if len(mask) != num_lists:
        raise ValueError(
            f"mask length {len(mask)} does not match number of lists {num_lists}"
        )

    sizes = segment_sizes(offsets)
    element_mask = np.repeat(mask, sizes)

    kept_sizes = sizes[mask]
    new_offsets = np.zeros(len(kept_sizes) + 1, dtype=np.int64)
    np.cumsum(kept_sizes, out=new_offsets[1:])

    return _rebuild(layout, new_offsets, content[element_mask])


def transform_lists(array, out, list_size, op):
    """Apply the n-ary `op` across the items of equal-size lists.

    Each list must contain exactly `list_size` items; `op` receives one
    argument per item position (each a NumPy array over all lists) and must
    return one value per list, e.g. ``lambda x, y, z: x + y + z``.
    The result is written into `out` (a NumPy array of length num_lists).
    """
    _, offsets, content = _listoffset_parts(array)
    num_segments = len(offsets) - 1

    columns = content[: num_segments * list_size].reshape(num_segments, list_size)
    result = op(*(columns[:, i] for i in range(list_size)))

    out_buffer = out if isinstance(out, np.ndarray) else np.asarray(out)
    out_buffer[:num_segments] = result
    return out
