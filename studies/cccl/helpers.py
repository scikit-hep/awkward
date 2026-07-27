import awkward as ak
import awkward as ak
import numpy as np
import cupy as cp
from cuda.compute import ZipIterator, PermutationIterator
import nvtx

from _segment_algorithms import segment_sizes, select_segments, segmented_select, transform_segments


@nvtx.annotate("empty_like")
def empty_like(array, kind="empty"):
    # Use low-level API to avoid dispatch and from_buffers overhead
    if isinstance(array, ak.Array):
        layout = array.layout
    elif hasattr(array, 'layout'):
        layout = array.layout
    elif isinstance(array, ak.contents.Content):
        layout = array
    else:
        layout = ak.to_layout(array)

    # Recursively copy the layout tree, allocating empty buffers for data
    def copy_with_empty_buffers(content):
        backend = content._backend
        xp = backend.nplike

        if isinstance(content, ak.contents.NumpyArray):
            # Allocate empty data buffer
            empty_data = xp.empty(content.data.shape, dtype=content.data.dtype)
            return ak.contents.NumpyArray(
                empty_data,
                parameters=content._parameters,
                backend=backend
            )
        elif isinstance(content, ak.contents.ListOffsetArray):
            # Copy offsets to avoid sharing buffers between arrays
            offsets_array = xp.asarray(content.offsets).copy()
            # Wrap in appropriate Index type
            if isinstance(content.offsets, ak.index.Index32):
                new_offsets = ak.index.Index32(offsets_array)
            elif isinstance(content.offsets, ak.index.IndexU32):
                new_offsets = ak.index.IndexU32(offsets_array)
            else:
                new_offsets = ak.index.Index64(offsets_array)

            return ak.contents.ListOffsetArray(
                new_offsets,
                copy_with_empty_buffers(content.content),
                parameters=content._parameters
            )
        elif isinstance(content, ak.contents.ListArray):
            # Copy starts/stops to avoid sharing buffers
            starts_array = xp.asarray(content.starts).copy()
            stops_array = xp.asarray(content.stops).copy()
            # Wrap in appropriate Index types
            if isinstance(content.starts, ak.index.Index32):
                new_starts = ak.index.Index32(starts_array)
                new_stops = ak.index.Index32(stops_array)
            elif isinstance(content.starts, ak.index.IndexU32):
                new_starts = ak.index.IndexU32(starts_array)
                new_stops = ak.index.IndexU32(stops_array)
            else:
                new_starts = ak.index.Index64(starts_array)
                new_stops = ak.index.Index64(stops_array)

            return ak.contents.ListArray(
                new_starts,
                new_stops,
                copy_with_empty_buffers(content.content),
                parameters=content._parameters
            )
        elif isinstance(content, ak.contents.RecordArray):
            return ak.contents.RecordArray(
                [copy_with_empty_buffers(c) for c in content.contents],
                content.fields,
                length=content.length,
                parameters=content._parameters,
                backend=backend
            )
        elif isinstance(content, ak.contents.IndexedArray):
            # Copy index to avoid sharing buffers
            index_array = xp.asarray(content.index).copy()
            # Wrap in appropriate Index type
            if isinstance(content.index, ak.index.Index32):
                new_index = ak.index.Index32(index_array)
            elif isinstance(content.index, ak.index.IndexU32):
                new_index = ak.index.IndexU32(index_array)
            else:
                new_index = ak.index.Index64(index_array)

            return ak.contents.IndexedArray(
                new_index,
                copy_with_empty_buffers(content.content),
                parameters=content._parameters
            )
        elif isinstance(content, ak.contents.IndexedOptionArray):
            # Copy index to avoid sharing buffers
            index_array = xp.asarray(content.index).copy()
            # Wrap in appropriate Index type
            if isinstance(content.index, ak.index.Index32):
                new_index = ak.index.Index32(index_array)
            elif isinstance(content.index, ak.index.IndexU32):
                new_index = ak.index.IndexU32(index_array)
            else:
                new_index = ak.index.Index64(index_array)

            return ak.contents.IndexedOptionArray(
                new_index,
                copy_with_empty_buffers(content.content),
                parameters=content._parameters
            )
        elif isinstance(content, ak.contents.RegularArray):
            return ak.contents.RegularArray(
                copy_with_empty_buffers(content.content),
                content.size,
                content.length,
                parameters=content._parameters
            )
        else:
            # For other types, fallback to copy
            return content

    new_layout = copy_with_empty_buffers(layout)
    return ak.Array(new_layout)


@nvtx.annotate("awkward_to_iterator")
def awkward_to_cccl_iterator(array=None, form=None, buffers=None, dtype=None, return_offsets=True):
    """
    Convert an Awkward Array to a cuda.compute iterator (zero-copy).

    This function recursively traverses the Awkward form structure and constructs
    the corresponding cuda.compute iterator:
    - NumpyArray -> CuPy array
    - RecordArray -> ZipIterator over field iterators
    - IndexedArray -> PermutationIterator with index buffer
    - ListOffsetArray -> Iterator for the flattened content

    The resulting iterator can be used with the cuda.compute library.

    Args:
        array: Awkward Array (if starting fresh)
        form: Awkward form (from ak.to_buffers)
        buffers: Buffer dict (from ak.to_buffers)
        dtype: Optional dtype to cast to (e.g., np.float32 for GPU structs)
        return_offsets: If True, extract and return offsets for list structures (default: True)

    Returns:
        Iterator or CuPy array representing the structure
        Dictionary with metadata: {"form": ..., "buffers": ..., "offsets": ..., "length": ..., "count": ...}
        - offsets: CuPy array of offsets if list structure exists, else None
        - length: Array length (number of lists)
        - count: Total number of items across all lists (avoids .get() calls)
    """
    # Initial call: extract form and buffers from array
    initial_call = form is None and buffers is None
    length = None

    if initial_call:
        if array is None:
            raise ValueError(
                "Must provide either 'array' or both 'form' and 'buffers'")

        # Fast path: use low-level API to avoid dispatch overhead
        # Access layout directly if it's an ak.Array, otherwise convert
        if isinstance(array, ak.Array):
            layout = array.layout
        elif hasattr(array, 'layout'):
            # It's a Record or similar
            layout = array.layout
        elif isinstance(array, ak.contents.Content):
            # Already a layout
            layout = array
        else:
            # Rare fallback: need to convert to layout (will use dispatch, but rare)
            layout = ak.to_layout(array)

        # Check if already on CUDA backend, if not convert using low-level method
        if layout._backend.name != "cuda":
            from awkward._backends.dispatch import regularize_backend
            cuda_backend = regularize_backend("cuda")
            layout = layout.to_backend(cuda_backend)

        # Use low-level to_buffers to avoid @high_level_function dispatch overhead
        form, length, buffers = ak._do.to_buffers(layout)

    # Helper to extract offsets from the form structure
    def extract_offsets_from_form(form_to_search):
        """Navigate through form structure to find and extract list offsets."""
        search_form = form_to_search

        # Unwrap IndexedForm/IndexedOptionForm
        if isinstance(search_form, (ak.forms.IndexedForm, ak.forms.IndexedOptionForm)):
            search_form = search_form.content

        # Unwrap RecordForm (use first field)
        if isinstance(search_form, ak.forms.RecordForm):
            search_form = search_form.contents[0]

        # Extract offsets from ListOffsetForm/ListForm
        if isinstance(search_form, (ak.forms.ListOffsetForm, ak.forms.ListForm)):
            offsets_key = f"{search_form.form_key}-offsets"
            return cp.asarray(buffers[offsets_key], dtype=np.int64)

        return None

    # Extract offsets if this is the initial call and offsets are requested
    offsets = None
    count = None
    if initial_call and return_offsets:
        offsets = extract_offsets_from_form(form)
        # Pre-compute count (total number of items in the list array)
        if offsets is not None:
            count = int(offsets[-1])

    # Helper to create metadata dict for return
    def make_metadata():
        return {
            "form": form,
            "buffers": buffers,
            "offsets": offsets,
            "length": length,
            "count": count
        }

    # Base case: NumpyArray - return the flat data buffer
    if isinstance(form, ak.forms.NumpyForm):
        data_key = f"{form.form_key}-data"
        buffer = buffers[data_key]
        if dtype is not None:
            buffer = cp.asarray(buffer, dtype=dtype)
        else:
            buffer = cp.asarray(buffer)

        if initial_call:
            return buffer, make_metadata()
        else:
            return buffer, (form, buffers)

    # RecordArray: create ZipIterator over all fields
    elif isinstance(form, ak.forms.RecordForm):
        field_iterators = []
        for field_form in form.contents:
            field_iter, _ = awkward_to_cccl_iterator(
                form=field_form, buffers=buffers, dtype=dtype, return_offsets=False
            )
            field_iterators.append(field_iter)

        result = ZipIterator(*field_iterators)
        if initial_call:
            return result, make_metadata()
        else:
            return result, (form, buffers)

    # IndexedArray: create PermutationIterator with index mapping
    elif isinstance(form, (ak.forms.IndexedForm, ak.forms.IndexedOptionForm)):
        index_key = f"{form.form_key}-index"
        index_buffer = cp.asarray(buffers[index_key])

        # Recursively get iterator for the content
        content_iter, _ = awkward_to_cccl_iterator(
            form=form.content, buffers=buffers, dtype=dtype, return_offsets=False
        )

        result = PermutationIterator(content_iter, index_buffer)
        if initial_call:
            return result, make_metadata()
        else:
            return result, (form, buffers)

    # ListOffsetArray: return iterator for the flattened content
    # (Offsets are extracted at the top level if this is an initial call)
    elif isinstance(form, (ak.forms.ListOffsetForm, ak.forms.ListForm)):
        # Recursively handle the content (which is already flattened)
        content_iter, _ = awkward_to_cccl_iterator(
            form=form.content, buffers=buffers, dtype=dtype, return_offsets=False
        )

        if initial_call:
            return content_iter, make_metadata()
        else:
            return content_iter, (form, buffers)

    else:
        raise NotImplementedError(
            f"Form type {type(form).__name__} not yet supported. "
            f"Please add support or file an issue."
        )


@nvtx.annotate("reconstruct_with_offsets")
def reconstruct_with_offsets(list_array, new_offsets):
    """
    Given a list array and new offsets representing for example
    a filtered view, reconstruct the list array with the new offsets.
    """

    if isinstance(list_array, ak.Array):
        layout = list_array.layout
    elif hasattr(list_array, 'layout'):
        layout = list_array.layout
    elif isinstance(list_array, ak.contents.Content):
        layout = list_array
    else:
        layout = ak.to_layout(list_array)

    # Wrap new_offsets in an Index if it's not already
    if not isinstance(new_offsets, ak.index.Index):
        # Determine the appropriate Index type based on dtype
        if hasattr(new_offsets, 'dtype'):
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
                new_offsets,
                sliced_content,
                parameters=content._parameters
            )
        elif isinstance(content, ak.contents.IndexedArray):
            # Recurse through indexed wrapper
            new_content = reconstruct_list(content.content, new_offsets)
            return ak.contents.IndexedArray(
                content.index,
                new_content,
                parameters=content._parameters
            )
        elif isinstance(content, ak.contents.IndexedOptionArray):
            # Recurse through indexed option wrapper
            new_content = reconstruct_list(content.content, new_offsets)
            return ak.contents.IndexedOptionArray(
                content.index,
                new_content,
                parameters=content._parameters
            )
        elif isinstance(content, ak.contents.RecordArray):
            # For records, reconstruct each field
            new_contents = [reconstruct_list(
                c, new_offsets) for c in content.contents]
            # Length should match the number of lists (offsets length - 1)
            new_length = len(new_offsets.data) - 1 if isinstance(new_offsets,
                                                                 ak.index.Index) else len(new_offsets) - 1
            return ak.contents.RecordArray(
                new_contents,
                content.fields,
                length=new_length,
                parameters=content._parameters,
                backend=content._backend
            )
        else:
            # Shouldn't reach here for typical list arrays
            return content

    new_layout = reconstruct_list(layout, new_offsets)
    return ak.Array(new_layout)


@nvtx.annotate("filter_lists")
def filter_lists(array, cond):
    it, meta = awkward_to_cccl_iterator(array)
    in_segments = meta["offsets"]
    out_array = empty_like(array)
    it_out, meta_out = awkward_to_cccl_iterator(out_array)
    out_segments = meta_out["offsets"]
    num_items = meta["count"]
    segmented_select(
        it,
        in_segments,
        it_out,
        out_segments,
        cond,
        num_items
    )
    return reconstruct_with_offsets(out_array, out_segments)


@nvtx.annotate("select_lists")
def select_lists(array, mask):
    data_in, meta = awkward_to_cccl_iterator(array)
    offsets_in = meta["offsets"]
    offsets_out = meta["offsets"]
    num_lists = meta["length"]
    num_elements = meta["count"]
    out_array = empty_like(array)
    data_out, meta = awkward_to_cccl_iterator(out_array)
    d_num_selected_out = cp.empty(2, np.int32)
    select_segments(
        data_in,
        offsets_in,
        mask,
        data_out,
        offsets_out,
        d_num_selected_out,
        num_elements,
        num_lists)
    num_elements_kept, num_lists_kept = d_num_selected_out
    offsets_out = offsets_out[:num_lists_kept+1]
    return reconstruct_with_offsets(out_array, offsets_out)


@nvtx.annotate("list_sizes")
def list_sizes(array):
    _, meta = awkward_to_cccl_iterator(array)
    return segment_sizes(meta["offsets"])


@nvtx.annotate("transform_lists")
def transform_lists(array, out_array, list_size, op):
    data_in, meta = awkward_to_cccl_iterator(array)
    data_out, _ = awkward_to_cccl_iterator(out_array)
    num_segments = meta["length"]
    transform_segments(data_in, data_out, list_size, op, num_segments)
    return out_array
