# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import awkward_cpp

from awkward._backends.backend import Backend, KernelKeyType
from awkward._backends.dispatch import register_backend
from awkward._kernels import NumpyKernel
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._typing import Final

np = NumpyMetadata.instance()
numpy = Numpy.instance()


@register_backend(Numpy)
class NumpyBackend(Backend):
    name: Final[str] = "cpu"

    _numpy: Numpy

    @property
    def nplike(self) -> Numpy:
        return self._numpy

    @property
    def index_nplike(self) -> Numpy:
        return self._numpy

    def __init__(self):
        self._numpy = Numpy.instance()

    def __getitem__(self, index: KernelKeyType) -> NumpyKernel:
        return NumpyKernel(awkward_cpp.cpu_kernels.kernel[index], index)
