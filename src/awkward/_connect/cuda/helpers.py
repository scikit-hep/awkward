# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import cupy as cp
import numpy as np

try:
    import nvtx
except ImportError:

    class nvtx:
        @staticmethod
        def annotate(*args, **kwargs):
            def deco(fn):
                return fn

            return deco


from cuda.compute import PermutationIterator, ZipIterator

import awkward as ak

# Layout utilities are backend-neutral and shared with the CPU backend.
from awkward._connect.lazy._layout import (
    empty_like,
    reconstruct_with_offsets,
)

from ._segment_algorithms import (
    segment_sizes,
    segmented_select,
    select_segments,
    transform_segments,
)


@nvtx.annotate("awkward_to_iterator")
def awkward_to_cccl_iterator(
    array=None, form=None, buffers=None, dtype=None, return_offsets=True
):
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
            raise ValueError("Must provide either 'array' or both 'form' and 'buffers'")

        # Fast path: use low-level API to avoid dispatch overhead
        # Access layout directly if it's an ak.Array, otherwise convert
        if isinstance(array, ak.Array):
            layout = array.layout
        elif hasattr(array, "layout"):
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
            "count": count,
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


@nvtx.annotate("filter_lists")
def filter_lists(array, cond):
    """Keep elements within each list for which the callable `cond` is true."""
    it, meta = awkward_to_cccl_iterator(array)
    in_segments = meta["offsets"]
    out_array = empty_like(array)
    it_out, meta_out = awkward_to_cccl_iterator(out_array)
    out_segments = meta_out["offsets"]
    num_items = meta["count"]
    segmented_select(it, in_segments, it_out, out_segments, cond, num_items)
    return reconstruct_with_offsets(out_array, out_segments)


@nvtx.annotate("select_lists")
def select_lists(array, mask):
    """Keep entire lists selected by the per-list `mask`."""
    data_in, meta_in = awkward_to_cccl_iterator(array)
    offsets_in = meta_in["offsets"]
    num_elements = meta_in["count"]

    # The output gets its own offsets buffer; writing into `offsets_in` here
    # would corrupt the input array in place (the buffers are zero-copy views
    # of the input layout).
    out_array = empty_like(array)
    data_out, meta_out = awkward_to_cccl_iterator(out_array)
    offsets_out = meta_out["offsets"]

    mask = cp.asarray(mask)
    if mask.dtype != cp.int8:
        mask = mask.astype(cp.int8)

    d_num_selected_out = cp.empty(2, np.int32)
    select_segments(
        data_in,
        offsets_in,
        mask,
        data_out,
        offsets_out,
        d_num_selected_out,
        num_elements,
    )
    num_lists_kept = int(d_num_selected_out[1])
    offsets_out = offsets_out[: num_lists_kept + 1]
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
