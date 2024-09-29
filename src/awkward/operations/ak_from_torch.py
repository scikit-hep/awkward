# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("from_torch",)


@high_level_function()
def from_torch(array):
    """
    Args:
        array: (PyTorch Tensor):
            Tensor to convert into an Awkward Array.

    Converts a PyTorch Tensor into an Awkward Array.

    If `array` contains any other data types the function raises an error.
    """

    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array)


def _impl(array):
    try:
        import torch
    except ImportError as err:
        raise ImportError(
            """to use ak.from_torch, you must install 'torch' package with:

         pip install torch

or

        conda install pytorch"""
        ) from err

    # check if array is a Tensor
    if not isinstance(array, torch.Tensor):
        raise TypeError("""only PyTorch Tensor can be converted to Awkward Array""")

    # keep the resulting array on the same device as input tensor
    device = "cuda" if array.is_cuda else "cpu"

    # convert tensors to cupy if they are on cuda
    if device == "cuda":
        from awkward._nplikes.cupy import Cupy

        cp = Cupy.instance()

        # zero-copy data exchange through DLPack
        cp_array = cp.from_dlpack(array)
        ak_array = ak.from_cupy(cp_array)

    else:
        np_array = array.numpy()
        ak_array = ak.from_numpy(np_array)

    return ak_array
