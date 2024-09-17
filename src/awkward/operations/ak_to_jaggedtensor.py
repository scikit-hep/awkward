# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("to_jaggedtensor",)


@high_level_function()
def to_jaggedtensor(array, padded=False, padding_value=0, max_lengths=None, keep_regular=True):
    """
    Args:
        array: Array-like data. May be a high level #ak.Array,
            or low-level #ak.contents.ListOffsetArray, #ak.contents.ListArray,
            #ak.contents.RegularArray, #ak.contents.NumpyArray
        padded (bool): if True, return a padded tensor using a `jagged_to_padded_dense` function
            from PyTorch; otherwise return a jagged tensor.
        padding_value (float): if `padded` = True, sets a value for padding.
        max_lengths (int[]): if `padded` = True, sets a length to be padded to, for each jagged dimension.
        keep_regular (bool): if True, tries to keep the regular structure in the output.
            If False, automatically converts all RegularArrays to ListOffsetArray.

    Converts `array` (only ListOffsetArray, ListArray, RegularArray and NumpyArray data types supported)
    into a jagged tensor, if possible. Jagged tensor structure looks like this: Tuple(torch.Tensor, List[torch.Tensor])

    If `array` contains any other data types (RecordArray for example) the function raises an error.
    """

    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, padded, padding_value, max_lengths, keep_regular)


def _impl(array, padded, padding_value, max_lengths, keep_regular):
    try:
        import torch
        import fbgemm_gpu
    except ImportError as err:
        raise ImportError(
            """to use ak.to_jaggedtensor, you must install 'torch' and 'fbgemm_gpu' packages with:

        pip install torch or conda install pytorch
        pip install fbgemm-gpu-cpu or pip install fbgemm-gpu"""
        ) from err

    # unwrap the awkward array if it was made with ak.Array function
    # also transforms a python list to awkward array
    array = ak.to_layout(array, allow_record=False)

    if isinstance(array, ak.contents.numpyarray.NumpyArray):
        return torch.tensor(array.data)
    else:

        if not padded and not (isinstance(array, ak.contents.regulararray.RegularArray)):
            flat_values, nested_row_splits = _recursive_call(array, [], keep_regular)

        else:
            # create a list of max lengths for each jagged dimension
            max_lengths = _count_max_lengths(array, max_lengths)
            flat_values, nested_row_splits = _recursive_call(array, [], keep_regular=False)

        # since "jagged_to_padded_dense" not implemented for '64-bit floating point' convert float64 -> float32
        if isinstance(flat_values.dtype, type(np.dtype(np.float64))):
            try:
                dense_test = torch.tensor([[[1, 1], [0, 0]], [[2, 2], [3, 3]]], dtype=torch.float64)
                offsets_test = torch.tensor([0, 1, 3], dtype=torch.float64)
                torch.ops.fbgemm.dense_to_jagged(dense_test, [offsets_test])
            except RuntimeError as error:
                raise error

        # convert numpy to a torch tensor
        dense = torch.from_numpy(flat_values)

        # convert a 'list of numpy' to a 'list of tensors'
        offsets = [torch.from_numpy(item) for item in nested_row_splits]

        if not padded and not (isinstance(array, ak.contents.regulararray.RegularArray)):
            return (dense, offsets)
        else:
            # create a padded dense tensor using torch function
            dense_tensor = torch.ops.fbgemm.jagged_to_padded_dense(dense, offsets, max_lengths, padding_value)
            return dense_tensor


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
                        " into a PyTorch JaggedTensor. Please use ak.from_regular() or set keep_regular = False"
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


def _count_max_lengths(array, max_lengths):
    if max_lengths is None:
        _, max_depth = array.minmax_depth
        max_lengths = []
        for i in range(1, max_depth):
            max_lengths.append(ak.max(ak.num(array, i)))
        return max_lengths
    else:
        return max_lengths