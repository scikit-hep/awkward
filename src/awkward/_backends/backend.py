# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import awkward as ak
from awkward._kernels import KernelError
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyLike, NumpyMetadata
from awkward._singleton import PublicSingleton
from awkward._typing import Callable, Tuple, TypeAlias, TypeVar

np = NumpyMetadata.instance()
numpy = Numpy.instance()


T_co = TypeVar("T_co", covariant=True)
KernelKeyType: TypeAlias = Tuple[Any, ...]
KernelType: TypeAlias = "Callable[..., KernelError | None]"


class Backend(PublicSingleton, ABC):
    name: str

    @property
    @abstractmethod
    def nplike(self) -> NumpyLike:
        raise NotImplementedError

    @property
    @abstractmethod
    def index_nplike(self) -> NumpyLike:
        raise NotImplementedError

    def __getitem__(self, key: KernelKeyType) -> KernelType:
        raise NotImplementedError

    def prepare_reducer(self, reducer: ak._reducers.Reducer) -> ak._reducers.Reducer:
        return reducer

    def format_kernel_error(
        self,
        error: KernelError,
    ):
        if error.filename is None:
            filename = ""
        else:
            filename = " (in compiled code: " + error.filename.decode(
                errors="surrogateescape"
            ).lstrip("\n").lstrip("(")

        assert error.str is not None
        message = error.str.decode(errors="surrogateescape")

        if error.attempt != ak._util.kSliceNone:
            message += f" while attempting to get index {error.attempt}"

        message += filename
        return message

    def maybe_kernel_error(self, error: KernelError | None):
        if error is None or error.str is None:
            return
        else:
            raise ValueError(self.format_kernel_error(error))
