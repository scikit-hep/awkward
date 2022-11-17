# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import annotations

from awkward.nplikes.numpy import ArrayModuleNumpyLike


class Cupy(ArrayModuleNumpyLike):
    @property
    def array_module(self):
        import cupy

        return cupy
