# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Backend-neutral layout utilities for the lazy IR.

These operate through ``layout._backend.nplike`` so they work for both the
CPU (NumPy) and CUDA (CuPy) backends.
"""

from __future__ import annotations

import numpy as np

import awkward as ak


def contains_list(layout) -> bool:
    """
    Args:
        layout (ak.contents.Content): Layout tree to inspect.

    Returns True if any node of the layout tree is a list type
    (``ListOffsetArray``, ``ListArray``, or ``RegularArray``).
    """
    stack = [layout]
    while stack:
        current = stack.pop()
        if isinstance(
            current,
            (
                ak.contents.ListOffsetArray,
                ak.contents.ListArray,
                ak.contents.RegularArray,
            ),
        ):
            return True
        if isinstance(current, ak.contents.RecordArray):
            stack.extend(current.contents)
        elif hasattr(current, "content"):
            stack.append(current.content)
    return False


def is_fusible_numeric_list(array) -> bool:
    """
    Args:
        array (ak.Array or ak.contents.Content): Array or layout to test.

    Returns True only for a plain ``var`` list of unparameterized numbers: a
    ``ListOffsetArray``/``ListArray`` (the codegens normalize the latter) whose
    content is a numeric ``NumpyArray`` with no behavior-bearing parameters.

    Fusion lowers raw element-wise arithmetic over a shared offsets buffer, so
    it is correct only for that shape; everything else must fall back to the
    eager interpreter to preserve exact semantics:

    - strings / bytestrings carry ``__array__`` parameters on the list and its
      char content, so fusing would compare/scale characters (silent wrong);
    - ``RegularArray`` would degrade ``N * M`` to ``N * var``;
    - ``IndexedArray`` / ``IndexedOptionArray`` misapply a list-level index to
      flat elements (silent wrong / out-of-bounds device reads);
    - records / custom parameters must round-trip through eager to be preserved.
    """
    layout = array.layout if isinstance(array, ak.Array) else array
    if not isinstance(layout, (ak.contents.ListOffsetArray, ak.contents.ListArray)):
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
    """Normalize and validate a layout before building a cuda.compute iterator.

    A flat content iterator plus one offsets buffer can only represent layouts
    whose list content is contiguous and in order, so:

    - ``IndexedOptionArray`` (missing values) has no iterator representation at
      all — its ``-1`` indices would be out-of-bounds reads;
    - an ``IndexedArray`` above a list permutes *lists*, which a permutation of
      the flattened elements cannot express, so it is projected (copies, but any
      zero-copy answer would be wrong); an ``IndexedArray`` directly over leaf
      data stays zero-copy as a ``PermutationIterator``;
    - top-level ``ListArray`` / ``RegularArray`` become ``ListOffsetArray64`` so
      offsets always exist; a ``ListOffsetArray`` is repacked only if it does
      not start at zero.

    Args:
        layout (ak.contents.Content): Layout to normalize.

    Returns the normalized ``ak.contents.Content``, safe to pass to
    ``awkward_to_cccl_iterator``.

    Raises:
        NotImplementedError: If the layout carries missing values
            (``IndexedOptionArray``), which have no iterator representation.
    """
    if isinstance(layout, ak.contents.IndexedOptionArray):
        raise NotImplementedError(
            "arrays with missing values (IndexedOptionArray) cannot be "
            "represented as a cuda.compute iterator"
        )
    if isinstance(layout, ak.contents.IndexedArray) and contains_list(layout.content):
        layout = layout.project()
    if isinstance(
        layout,
        (
            ak.contents.ListArray,
            ak.contents.RegularArray,
            ak.contents.ListOffsetArray,
        ),
    ):
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
    """
    Args:
        prototype (ak.index.Index): Index whose subclass should be matched.
        data: Buffer to wrap.

    Returns ``data`` wrapped in the same ``ak.index.Index`` subclass as
    ``prototype``.
    """
    if isinstance(prototype, ak.index.Index32):
        return ak.index.Index32(data)
    elif isinstance(prototype, ak.index.IndexU32):
        return ak.index.IndexU32(data)
    else:
        return ak.index.Index64(data)


def empty_like(array, kind="empty"):
    """Copy a layout tree, allocating uninitialized data buffers.

    Offsets/starts/stops/index buffers are copied (not shared) so writes to the
    result never alias the input; data buffers are freshly allocated and
    uninitialized.

    Args:
        array (ak.Array or ak.contents.Content): Array whose structure to copy.
        kind (str): Reserved for future buffer-fill modes; currently unused.

    Returns an ``ak.Array`` with the same structure and copied index buffers but
    uninitialized data.

    Raises:
        NotImplementedError: If a layout node type is not supported (copying it
            would risk aliasing the input's buffers).
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
    Args:
        list_array (ak.Array or ak.contents.Content): The list array whose
            top-level offsets should be replaced.
        new_offsets: New offsets (an ``ak.index.Index`` or a buffer) representing
            for example a filtered view; the content is sliced to match.

    Returns an ``ak.Array`` reconstructed with ``new_offsets``, descending
    through indexed/record wrappers to find the top-level list.
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
