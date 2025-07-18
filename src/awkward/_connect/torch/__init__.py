# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import torch

from awkward._connect.torch.reducers import get_torch_reducer  # noqa: F401

def get_torch_ufunc(ufunc):
    return getattr(torch.numpy, ufunc.__name__, ufunc)
