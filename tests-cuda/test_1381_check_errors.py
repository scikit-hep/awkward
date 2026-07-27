# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import traceback

import numpy as np  # noqa: F401
import pytest

import awkward as ak
import awkward._connect.cuda

try:
    ak.numba.register_and_check()
except ImportError:
    pytest.skip(reason="too old Numba version", allow_module_level=True)


def test():
    v2_array = ak.Array([1, 2, 3, 4, 5]).layout

    v2_array_cuda = ak.to_backend(v2_array, "cuda")

    with pytest.raises(ValueError) as err:
        v2_array_cuda[10,]

    assert isinstance(err.value, ValueError)

    message = "".join(traceback.format_exception(err.type, err.value, err.tb))
    assert (
        "ValueError: index out of range in compiled CUDA code "
        "(awkward_RegularArray_getitem_next_at)\n"
    ) in message
