# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward_cpp

import awkward as ak
from awkward._backends.backend import Backend, KernelKeyType
from awkward._backends.dispatch import register_backend
from awkward._kernels import TorchKernel
from awkward._nplikes.torch import Torch
from awkward._typing import Final


@register_backend(Torch)  # type: ignore[type-abstract]
class TorchBackend(Backend):
    name: Final[str] = "torch"

    _torch: Torch

    @property
    def nplike(self) -> Torch:
        return self._torch

    def __init__(self):
        self._torch = Torch.instance()

    def __getitem__(self, index: KernelKeyType) -> TorchKernel:
        # Torch uses Awkward's C++ kernels for index-only operations
        return TorchKernel(awkward_cpp.cpu_kernels.kernel[index], index)

    def prepare_reducer(self, reducer: ak._reducers.Reducer) -> ak._reducers.Reducer:
        from awkward._connect.torch import get_torch_reducer

        return get_torch_reducer(reducer)
