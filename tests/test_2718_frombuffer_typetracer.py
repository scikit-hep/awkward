# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

import awkward as ak
from awkward._nplikes.typetracer import TypeTracer

typetracer = TypeTracer.instance()


def test():
    buffer = b"Hello world, what's the news?"
    left = np.frombuffer(buffer, np.float32, 4)
    right = typetracer.frombuffer(buffer, dtype=np.float32, count=4)
    assert left.dtype == right.dtype
    assert left.shape == right.shape


def test_zeros_like_numeric():
    source = ak.Array([[1, 2, 3]], backend="typetracer")
    result = ak.zeros_like(source)

    # Known lengths and dtypes should match
    assert result.layout.content.dtype == source.layout.content.dtype
    assert result.layout.content.length == source.layout.content.length


def test_zeros_like_strings():
    # Check strings!
    source = ak.Array(["hello", "world"], backend="typetracer")
    result = ak.zeros_like(source)
    assert result.layout.content.dtype == source.layout.content.dtype
    # zeros_like for strings means empty-string!
    assert result.layout.content.length == 0
