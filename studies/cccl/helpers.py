import awkward as ak
import awkward as ak
import numpy as np
import cupy as cp
from cuda.compute import ZipIterator, PermutationIterator


def empty_like(array, kind="empty"):
    form, length, bufs = ak.to_buffers(array)
    backend = ak.backend(array)
    xp = __import__("cupy" if backend == "cuda" else "numpy")
    new_bufs = {
        key: xp.empty_like(buf) if key.endswith("data") else buf
        for key, buf in bufs.items()
    }
    return ak.from_buffers(form, length, new_bufs, backend=backend)


def awkward_to_cccl_iterator(array=None, form=None, buffers=None, dtype=None, return_offsets=True):
    """
    Convert an Awkward Array to a CCCL iterator (zero-copy).

    This function recursively traverses the Awkward form structure and constructs
    the corresponding CCCL iterator:
    - NumpyArray -> CuPy buffer
    - RecordArray -> ZipIterator over field iterators
    - IndexedArray -> PermutationIterator with index buffer
    - ListOffsetArray -> Iterator for the flattened content

    The result is a CCCL iterator that provides zero-copy access to GPU buffers,
    with automatic handling of indexing, records, and jagged array flattening.

    Example usage:
        # Convert an Awkward Array with structure: IndexedArray -> RecordArray -> ...
        iterator, metadata = awkward_to_cccl_iterator(my_array, dtype=np.float32)
        # metadata = {"form": ..., "buffers": ..., "offsets": ..., "length": ...}
        # Now `iterator` can be used directly with CCCL algorithms
        # For application-specific indexing, wrap with additional PermutationIterators

    Args:
        array: Awkward Array (if starting fresh)
        form: Awkward form (from ak.to_buffers)
        buffers: Buffer dict (from ak.to_buffers)
        dtype: Optional dtype to cast to (e.g., np.float32 for GPU structs)
        return_offsets: If True, extract and return offsets for list structures (default: True)

    Returns:
        CCCL iterator or CuPy array representing the structure
        Dictionary with metadata: {"form": ..., "buffers": ..., "offsets": ..., "length": ...}
        - offsets: CuPy array of offsets if list structure exists, else None
        - length: Array length from ak.to_buffers
    """
    # Initial call: extract form and buffers from array
    initial_call = form is None and buffers is None
    length = None

    if initial_call:
        if array is None:
            raise ValueError(
                "Must provide either 'array' or both 'form' and 'buffers'")
        array_gpu = ak.to_backend(array, "cuda")
        form, length, buffers = ak.to_buffers(array_gpu)

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
    if initial_call and return_offsets:
        offsets = extract_offsets_from_form(form)

    # Helper to create metadata dict for return
    def make_metadata():
        return {
            "form": form,
            "buffers": buffers,
            "offsets": offsets,
            "length": length
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

