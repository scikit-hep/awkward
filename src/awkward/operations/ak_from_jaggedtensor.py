# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("from_jaggedtensor",)


@high_level_function()
def from_jaggedtensor(array):
    """
    Args:
        array: (PyTorch JaggedTensor):
            JaggedTensor to convert into an Awkward Array. The data type of a PyTorch JaggedTensor
            is a 2-tuple of a `torch.Tensor` and a list of `torch.Tensors`.

    Converts a PyTorch JaggedTensor into an Awkward Array.

    If `array` contains any other data types the function raises an error.
    """

    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array)


def _impl(array):
    # keep the resulting array on the same device as input tensor
    device = "cuda" if array[0].is_cuda else "cpu"

    # convert tensors to cupy if they are on cuda
    if device == "cuda":
        try:
            from awkward._nplikes.cupy import Cupy

            cp = Cupy.instance()
        except (ModuleNotFoundError, ImportError) as err:
            raise err
        content_cp = cp.asarray(array[0])
        content = ak.contents.NumpyArray(content_cp)

        offsets_arr = []
        for offset in array[1]:
            offset_cp = cp.asarray(offset)
            offsets_arr.append(ak.index.Index64(offset_cp))
    else:
        content = ak.contents.NumpyArray(array[0])

        offsets_arr = []
        for offset in array[1]:
            offsets_arr.append(ak.index.Index64(offset))

    # if a tensor has one *ragged dimension*
    if len(offsets_arr) == 1:
        result = ak.contents.ListOffsetArray(offsets_arr[0], content)
        return ak.Array(result)

    # if a tensor has multiple *ragged dimensions*
    return ak.Array(_recursive_call(content, offsets_arr, 0, device))


def _recursive_call(content, offsets_arr, count, device):
    if count == len(offsets_arr) - 2:
        return ak.contents.ListOffsetArray(
            offsets_arr[count],
            ak.contents.ListOffsetArray(offsets_arr[count + 1], content),
        )
    else:
        return ak.contents.ListOffsetArray(
            offsets_arr[count], _recursive_call(content, offsets_arr, count)
        )
