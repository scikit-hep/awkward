# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Backend-neutral layout utilities for the lazy IR.

These operate through ``layout._backend.nplike`` so they work for both the
CPU (NumPy) and CUDA (CuPy) backends.
"""

from __future__ import annotations

import numpy as np

import awkward as ak


def is_fusible_numeric_list(array) -> bool:
    """True only for a plain ``var`` list of unparameterized numbers.

    Fusion lowers raw element-wise arithmetic over a shared offsets buffer, so
    it is correct **only** for a ``ListOffsetArray`` whose content is a numeric
    ``NumpyArray`` with no behavior-bearing parameters.  Everything else must
    fall back to the eager interpreter to preserve exact semantics:

    - strings / bytestrings carry ``__array__`` parameters on the list and its
      char content -> fusing would compare/scale characters (silent wrong);
    - ``RegularArray`` would degrade ``N * M`` to ``N * var``;
    - ``ListArray`` uses starts/stops (the CUDA offsets extractor only knows
      ``-offsets``), and ``IndexedArray`` / ``IndexedOptionArray`` misapply a
      list-level index to flat elements (silent wrong / OOB device reads);
    - records / custom parameters must round-trip through eager to be preserved.
    """
    layout = array.layout if isinstance(array, ak.Array) else array
    if not isinstance(layout, ak.contents.ListOffsetArray):
        return False
    if layout.parameters:
        return False
    content = layout.content
    if not isinstance(content, ak.contents.NumpyArray):
        return False
    if content.parameters:
        return False
    # Real numbers only (exclude datetime/timedelta/complex/object/char dtypes).
    return np.dtype(content.dtype).kind in "biuf"


def validate_iterator_layout(layout):
    """Normalize/validate a layout before building a cuda.compute iterator.

    Returns a layout safe for ``awkward_to_cccl_iterator``: top-level
    ``ListArray`` / ``RegularArray`` are normalized to ``ListOffsetArray`` (so
    offsets always exist), and layouts the recursive builder cannot represent
    without silent corruption raise ``NotImplementedError`` instead of
    returning wrong data or reading out of bounds.

    In particular an ``IndexedArray`` / ``IndexedOptionArray`` *wrapping a list*
    permutes list-level indices onto flat elements (and ``-1``/``None`` indices
    read out of bounds); such layouts must be packed first.
    """
    # Reject index-of-list before it silently corrupts (or OOB-reads).
    if isinstance(layout, (ak.contents.IndexedArray, ak.contents.IndexedOptionArray)):
        inner = layout.content
        if isinstance(
            inner,
            (
                ak.contents.ListOffsetArray,
                ak.contents.ListArray,
                ak.contents.RegularArray,
            ),
        ):
            raise NotImplementedError(
                "awkward_to_cccl_iterator does not support an "
                f"{type(layout).__name__} wrapping a list; call ak.to_packed() "
                "(or ak.enforce_type) to materialize it first"
            )
    # Normalize a top-level list to ListOffsetArray so offsets always exist.
    if isinstance(layout, (ak.contents.ListArray, ak.contents.RegularArray)):
        return layout.to_ListOffsetArray64(True)
    return layout


def _as_layout(array):
    if isinstance(array, ak.Array):
        return array.layout
    elif hasattr(array, "layout"):
        return array.layout
    elif isinstance(array, ak.contents.Content):
        return array
    else:
        return ak.to_layout(array)


def _wrap_index_like(prototype, data):
    """Wrap `data` in the same ak.index.Index subclass as `prototype`."""
    if isinstance(prototype, ak.index.Index32):
        return ak.index.Index32(data)
    elif isinstance(prototype, ak.index.IndexU32):
        return ak.index.IndexU32(data)
    else:
        return ak.index.Index64(data)


def empty_like(array, kind="empty"):
    """Copy a layout tree, allocating uninitialized data buffers.

    Offsets/starts/stops/index buffers are *copied* (not shared) so writes to
    the result never alias the input.  Data buffers are freshly allocated and
    uninitialized.
    """
    layout = _as_layout(array)

    # Recursively copy the layout tree, allocating empty buffers for data
    def copy_with_empty_buffers(content):
        backend = content._backend
        xp = backend.nplike

        if isinstance(content, ak.contents.NumpyArray):
            # Allocate empty data buffer
            empty_data = xp.empty(content.data.shape, dtype=content.data.dtype)
            return ak.contents.NumpyArray(
                empty_data, parameters=content._parameters, backend=backend
            )
        elif isinstance(content, ak.contents.ListOffsetArray):
            # Copy offsets to avoid sharing buffers between arrays
            offsets_array = xp.asarray(content.offsets).copy()
            new_offsets = _wrap_index_like(content.offsets, offsets_array)

            return ak.contents.ListOffsetArray(
                new_offsets,
                copy_with_empty_buffers(content.content),
                parameters=content._parameters,
            )
        elif isinstance(content, ak.contents.ListArray):
            # Copy starts/stops to avoid sharing buffers
            starts_array = xp.asarray(content.starts).copy()
            stops_array = xp.asarray(content.stops).copy()
            new_starts = _wrap_index_like(content.starts, starts_array)
            new_stops = _wrap_index_like(content.stops, stops_array)

            return ak.contents.ListArray(
                new_starts,
                new_stops,
                copy_with_empty_buffers(content.content),
                parameters=content._parameters,
            )
        elif isinstance(content, ak.contents.RecordArray):
            return ak.contents.RecordArray(
                [copy_with_empty_buffers(c) for c in content.contents],
                content.fields,
                length=content.length,
                parameters=content._parameters,
                backend=backend,
            )
        elif isinstance(content, ak.contents.IndexedArray):
            # Copy index to avoid sharing buffers
            index_array = xp.asarray(content.index).copy()
            new_index = _wrap_index_like(content.index, index_array)

            return ak.contents.IndexedArray(
                new_index,
                copy_with_empty_buffers(content.content),
                parameters=content._parameters,
            )
        elif isinstance(content, ak.contents.IndexedOptionArray):
            # Copy index to avoid sharing buffers
            index_array = xp.asarray(content.index).copy()
            new_index = _wrap_index_like(content.index, index_array)

            return ak.contents.IndexedOptionArray(
                new_index,
                copy_with_empty_buffers(content.content),
                parameters=content._parameters,
            )
        elif isinstance(content, ak.contents.RegularArray):
            return ak.contents.RegularArray(
                copy_with_empty_buffers(content.content),
                content.size,
                content.length,
                parameters=content._parameters,
            )
        else:
            raise NotImplementedError(
                f"empty_like does not support {type(content).__name__} layouts; "
                f"sharing the original buffers here would risk in-place "
                f"corruption of the input"
            )

    new_layout = copy_with_empty_buffers(layout)
    return ak.Array(new_layout)


def reconstruct_with_offsets(list_array, new_offsets):
    """
    Given a list array and new offsets representing for example
    a filtered view, reconstruct the list array with the new offsets.
    """
    layout = _as_layout(list_array)

    # Wrap new_offsets in an Index if it's not already
    if not isinstance(new_offsets, ak.index.Index):
        # Determine the appropriate Index type based on dtype
        if hasattr(new_offsets, "dtype"):
            dtype = new_offsets.dtype
        else:
            dtype = np.int64

        if dtype == np.int32:
            new_offsets = ak.index.Index32(new_offsets)
        elif dtype == np.uint32:
            new_offsets = ak.index.IndexU32(new_offsets)
        else:
            new_offsets = ak.index.Index64(new_offsets)

    # Find the top-level list and reconstruct with new offsets
    def reconstruct_list(content, new_offsets):
        if isinstance(content, ak.contents.ListOffsetArray):
            # Slice content to match new offsets
            num_data = int(new_offsets.data[-1])
            sliced_content = content.content[:num_data]
            return ak.contents.ListOffsetArray(
                new_offsets, sliced_content, parameters=content._parameters
            )
        elif isinstance(content, ak.contents.IndexedArray):
            # Recurse through indexed wrapper
            new_content = reconstruct_list(content.content, new_offsets)
            return ak.contents.IndexedArray(
                content.index, new_content, parameters=content._parameters
            )
        elif isinstance(content, ak.contents.IndexedOptionArray):
            # Recurse through indexed option wrapper
            new_content = reconstruct_list(content.content, new_offsets)
            return ak.contents.IndexedOptionArray(
                content.index, new_content, parameters=content._parameters
            )
        elif isinstance(content, ak.contents.RecordArray):
            # For records, reconstruct each field
            new_contents = [reconstruct_list(c, new_offsets) for c in content.contents]
            # Length should match the number of lists (offsets length - 1)
            new_length = (
                len(new_offsets.data) - 1
                if isinstance(new_offsets, ak.index.Index)
                else len(new_offsets) - 1
            )
            return ak.contents.RecordArray(
                new_contents,
                content.fields,
                length=new_length,
                parameters=content._parameters,
                backend=content._backend,
            )
        else:
            # Shouldn't reach here for typical list arrays
            return content

    new_layout = reconstruct_list(layout, new_offsets)
    return ak.Array(new_layout)
