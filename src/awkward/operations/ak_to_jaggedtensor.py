# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("to_jaggedtensor",)

numpy = Numpy.instance()
np = NumpyMetadata.instance()


@high_level_function()
def to_jaggedtensor(
    array,
    backend=None,
):
    """
    Args:
        array: Array-like data. May be a high level #ak.Array,
            or low-level #ak.contents.ListOffsetArray, #ak.contents.ListArray,
            #ak.contents.RegularArray, #ak.contents.NumpyArray
        backend (None, `"cpu"`, `"cuda"`): If `"cpu"`, the `array` will be placed in
            main memory; if `"cuda"` the `array` will be placed in GPU global memory using CUDA;
            if None the backend of the `array` will be preserved.

    Converts `array` (only ListOffsetArray, ListArray, RegularArray and NumpyArray data types supported)
    into a PyTorch "jagged tensor", if possible. The data type of a PyTorch "jagged tensor" is a 2-tuple of a `torch.Tensor` and a list of `torch.Tensors`.
    The first `torch.Tensor` is the numerical contents of the array and the list of integer-valued `torch.Tensors` are offsets indicating where variable-length lists start and end.

    If `array` contains any other data types (RecordArray for example) the function raises a TypeError.
    """

    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, backend)


def _impl(array, backend):
    try:
        import torch
    except ImportError as err:
        raise ImportError(
            """to use ak.to_jaggedtensor, you must install 'torch' package with:

        pip install torch or conda install pytorch"""
        ) from err

    # unwrap the awkward array if it was made with ak.Array function
    # also transforms a python list to awkward array
    array = ak.to_layout(array, allow_record=False)

    # keep the resulting tensor on the same device as input
    device = ak.backend(array)
    if backend is not None:
        device = torch.device(backend)

    if isinstance(array, ak.contents.numpyarray.NumpyArray):
        return torch.tensor(array.data)
    elif isinstance(array, ak.contents.regulararray.RegularArray):
        # since a jagged tensor can't function with an empty offsets array
        raise TypeError(
            "RegularArrays cannot be converted into a PyTorch JaggedTensor. Try using ak.from_regular() if you still want to use this function."
        )
    else:
        flat_values, nested_row_splits = _recursive_call(array, [], keep_regular=True)

        # since "jagged_to_padded_dense" not implemented for '64-bit floating point' convert float64 -> float32
        if isinstance(flat_values.dtype, type(np.dtype(np.float64))):
            try:
                dense_test = torch.tensor(
                    [[[1, 1], [0, 0]], [[2, 2], [3, 3]]], dtype=torch.float64
                )
                offsets_test = torch.tensor([0, 1, 3], dtype=torch.float64)
                torch.ops.fbgemm.dense_to_jagged(dense_test, [offsets_test])
            except RuntimeError as error:
                raise error

        # check if cupy or numpy
        if isinstance(flat_values, np.ndarray):
            # convert numpy to a torch tensor
            dense = torch.from_numpy(flat_values).to(device)
            # convert a 'list of numpy' to a 'list of tensors'
            offsets = [torch.from_numpy(item).to(device) for item in nested_row_splits]
        else:
            # if cupy
            dense = torch.as_tensor(flat_values, device=device)
            offsets = [
                torch.as_tensor(item, device=device) for item in nested_row_splits
            ]

        return (dense, offsets)


def _recursive_call(layout, offsets_arr, keep_regular):
    try:
        # change all the possible layout types to ListOffsetArray
        if isinstance(layout, ak.contents.listarray.ListArray):
            layout = layout.to_ListOffsetArray64()
        elif isinstance(layout, ak.contents.regulararray.RegularArray):
            if keep_regular:
                # if RegularArray does not contain ListArrays or ListOffsetArrays return NumpyArray and accumulated offsets
                numpy_arr = layout.maybe_to_NumpyArray()
                if numpy_arr is not None:
                    return ak.to_numpy(numpy_arr), offsets_arr
                else:
                    raise TypeError(
                        "RegularArrays containing ListArray or ListOffsetArray cannot be converted"
                        " into a PyTorch JaggedTensor. Try using ak.from_regular() if you still want to use this function."
                    )
            layout = layout.to_ListOffsetArray64()
        elif not isinstance(
            layout,
            (
                ak.contents.listoffsetarray.ListOffsetArray,
                ak.contents.numpyarray.NumpyArray,
            ),
        ):
            raise TypeError(
                "Only arrays containing variable-length lists (var *) or"
                " regular-length lists (# *) of numbers can be converted into a PyTorch JaggedTensor"
            )

        # recursively gather all of the offsets of an array
        offsets_arr.append(layout.offsets.data)

    except AttributeError:
        # at the last iteration form a ragged tensor from the
        # accumulated offsets and flattened values of the array
        return layout.data, offsets_arr
    return _recursive_call(layout.content, offsets_arr, keep_regular)
