# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("to_torch",)

np = NumpyMetadata.instance()


@high_level_function()
def to_torch(array):
    """
    Args:
        array: Array-like data. May be a high level #ak.Array,
            or low-level #ak.contents.ListOffsetArray, #ak.contents.ListArray,
            #ak.contents.RegularArray, #ak.contents.NumpyArray

    Converts `array` (only ListOffsetArray, ListArray, RegularArray and NumpyArray data types supported)
    into a PyTorch Tensor, if possible.

    If `array` contains any other data types (RecordArray for example) the function raises a TypeError.
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
            """to use ak.to_torch, you must install 'torch' package with:

         pip install torch

or

        conda install pytorch"""
        ) from err

    # useful function that handles all possible input arrays
    array = ak.to_layout(array, allow_record=False)

    # get the device array is on
    device = ak.backend(array)

    if device not in ["cuda", "cpu"]:
        raise ValueError("Only 'cpu' and 'cuda' backend conversions are allowed")

    # convert to numpy or cupy if `array` on gpu
    try:
        backend_array = array.to_backend_array(allow_missing=False)
    except ValueError as err:
        raise TypeError(
            "Only arrays containing equal-length lists of numbers can be converted into a PyTorch Tensor"
        ) from err

    # check if cupy or numpy
    if isinstance(backend_array, np.ndarray):
        # convert numpy to a torch tensor
        tensor = torch.from_numpy(backend_array)
    else:
        # cupy -> torch tensor
        tensor = torch.utils.dlpack.from_dlpack(backend_array.toDlpack())

    return tensor
