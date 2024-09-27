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

        pip install torch or conda install pytorch"""
        ) from err

    # useful function that handles all possible input arrays
    array = ak.to_layout(array, allow_record=False)

    # get the device array is on
    device = ak.backend(array)

    # convert to numpy or cupy if `array` on gpu
    try:
        np_array = array.to_backend_array(allow_missing=False)
    except ValueError as err:
        raise TypeError(
            "Only arrays containing regular-length lists (# *) of numbers can be converted into a PyTorch Tensor"
        ) from err

    # check if cupy or numpy
    if isinstance(np_array, np.ndarray):
        # convert numpy to a torch tensor
        tensor = torch.from_numpy(np_array).to(device)
    else:
        # cupy -> torch tensor
        tensor = torch.as_tensor(np_array, device=device)

    return tensor
