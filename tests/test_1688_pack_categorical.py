# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak
from awkward._nplikes.numpy import Numpy

numpy = Numpy.instance()


def test():
    pytest.importorskip("pyarrow")

    this = ak.str.to_categorical(["one", "two", "one", "three", "one", "four"])
    assert ak.is_categorical(this)
    # Ensure packing by itself doesn't change the type
    this_packed = ak.to_packed(this)
    assert this_packed.type == this.type
    # Ensure the categories match between the two
    assert ak.all(ak.categories(this_packed) == ak.categories(this))

    # Ensure the inner types match (ignoring the length change)
    this_subset_packed = ak.to_packed(this[:-1])
    assert ak.is_categorical(this_subset_packed)
    assert this_subset_packed.type.content == this.type.content
    # Ensure the categories match between the two
    assert ak.all(ak.categories(this_subset_packed) == ak.categories(this))
