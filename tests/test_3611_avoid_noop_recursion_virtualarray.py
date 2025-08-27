# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
#
from __future__ import annotations

import sys

import numpy as np

from awkward._nplikes.numpy import Numpy
from awkward._nplikes.virtual import VirtualNDArray


def test_getitem():
    numpy_like = Numpy.instance()
    vc = VirtualNDArray(
        numpy_like,
        shape=(1,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1], dtype=np.int64),
    )
    for _ in range(sys.getrecursionlimit() + 1):
        vc = vc[:]
    assert vc.materialize()


def test_view():
    numpy_like = Numpy.instance()
    vc = VirtualNDArray(
        numpy_like,
        shape=(1,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1], dtype=np.int64),
    )
    for _ in range(sys.getrecursionlimit() + 1):
        vc = vc.view(np.dtype(np.int64))
    assert vc.materialize()


def test_transpose():
    numpy_like = Numpy.instance()
    vc = VirtualNDArray(
        numpy_like,
        shape=(1,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1], dtype=np.int64),
    )
    for _ in range(sys.getrecursionlimit() + 1):
        vc = vc.T
    assert vc.materialize()


def test_reshape():
    numpy_like = Numpy.instance()
    vc = VirtualNDArray(
        numpy_like,
        shape=(1,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1], dtype=np.int64),
    )
    for _ in range(sys.getrecursionlimit() + 1):
        vc = numpy_like.reshape(vc, (1,), copy=False)
    assert vc.materialize()


def test_asarray():
    numpy_like = Numpy.instance()

    # copy=False
    vc = VirtualNDArray(
        numpy_like,
        shape=(1,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1], dtype=np.int64),
    )
    for _ in range(sys.getrecursionlimit() + 1):
        vc = numpy_like.asarray(vc, dtype=np.dtype(np.int64), copy=False)
    assert vc.materialize()

    # copy=None
    vc = VirtualNDArray(
        numpy_like,
        shape=(1,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1], dtype=np.int64),
    )
    for _ in range(sys.getrecursionlimit() + 1):
        vc = numpy_like.asarray(vc, dtype=np.dtype(np.int64), copy=None)
    assert vc.materialize()
