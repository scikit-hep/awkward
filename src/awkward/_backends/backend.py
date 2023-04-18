# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from abc import ABC, abstractmethod

import awkward as ak
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpylike import NumpyLike, NumpyMetadata
from awkward._singleton import Singleton
from awkward._typing import Callable, Tuple, TypeAlias, TypeVar, Unpack

np = NumpyMetadata.instance()
numpy = Numpy.instance()


T = TypeVar("T", covariant=True)
KernelKeyType: TypeAlias = Tuple[str, Unpack[Tuple[np.dtype, ...]]]
KernelType: TypeAlias = Callable[..., None]


class Backend(Singleton, ABC):
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

    def apply_reducer(
        self,
        reducer: ak._reducers.Reducer,
        layout: ak.contents.NumpyArray,
        parents: ak.index.Index,
        outlength: int,
    ) -> ak.contents.NumpyArray:
        return reducer.apply(layout, parents, outlength)

    def apply_ufunc(self, ufunc, method, args, kwargs):
        return getattr(ufunc, method)(*args, **kwargs)
