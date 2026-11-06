# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.numpy import Numpy

REDUCERS = [
    ak.all,
    ak.any,
    ak.sum,
    ak.prod,
    ak.count,
    ak.count_nonzero,
    ak.min,
    ak.max,
    ak.argmin,
    ak.argmax,
]


@pytest.fixture
def fill_empty(monkeypatch):
    # make reads of unwritten entries fail on every run
    empty = Numpy.empty

    def filled(self, shape, *, dtype=None):
        out = empty(self, shape, dtype=dtype)
        if out.dtype.kind == "i":
            out[...] = np.iinfo(out.dtype).max
        return out

    monkeypatch.setattr(Numpy, "empty", filled)


def var():
    return ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2, 3, 6], dtype=np.int64)),
        ak.contents.NumpyArray(np.arange(6, dtype=np.int64)),
    )


def leaf():
    return ak.contents.NumpyArray(np.array([3, 1, 2], dtype=np.int64))


# every content has three rows so size=2 leaves the last one untrimmed
CONTENTS = {
    "var": var(),
    "regular": ak.contents.RegularArray(
        ak.contents.NumpyArray(np.arange(6, dtype=np.int64)), size=2
    ),
    "string": ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 3, 6], dtype=np.int64)),
        ak.contents.NumpyArray(
            np.frombuffer(b"abbccc", dtype=np.uint8), parameters={"__array__": "char"}
        ),
        parameters={"__array__": "string"},
    ),
    "indexed-option": ak.contents.IndexedOptionArray(
        ak.index.Index64(np.array([0, -1, 2], dtype=np.int64)), leaf()
    ),
    "byte-masked": ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([1, 0, 1], dtype=np.int8)), leaf(), valid_when=True
    ),
    "bit-masked": ak.contents.BitMaskedArray(
        ak.index.IndexU8(np.array([0b101], dtype=np.uint8)),
        leaf(),
        valid_when=True,
        length=3,
        lsb_order=True,
    ),
    "option-over-list": ak.contents.IndexedOptionArray(
        ak.index.Index64(np.array([0, -1, 2], dtype=np.int64)), var()
    ),
    "indexed": ak.contents.IndexedArray(
        ak.index.Index64(np.array([2, 1, 0], dtype=np.int64)), leaf()
    ),
    "leaf": leaf(),
}

LAYOUTS = {name: ak.contents.RegularArray(c, size=2) for name, c in CONTENTS.items()}
LAYOUTS["size-zero"] = ak.contents.RegularArray(var(), size=0, zeros_length=2)
LAYOUTS["nested"] = ak.contents.RegularArray(
    ak.contents.RegularArray(var(), size=1), size=2
)
LAYOUTS["list-above"] = ak.contents.ListOffsetArray(
    ak.index.Index64(np.array([0, 1, 1], dtype=np.int64)), LAYOUTS["var"]
)
LAYOUTS["option-above"] = ak.contents.UnmaskedArray(LAYOUTS["var"])


def check(layout, reducer, axis):
    array = ak.Array(layout)
    result = reducer(array, axis=axis)
    expected = reducer(ak.to_packed(array), axis=axis)
    assert result.to_list() == expected.to_list()
    assert result.type == expected.type


@pytest.mark.parametrize("layout", LAYOUTS.values(), ids=LAYOUTS.keys())
def test_reducers(fill_empty, layout):
    assert ak.validity_error(layout) == ""
    for reducer in REDUCERS:
        for axis in range(layout.minmax_depth[1]):
            check(layout, reducer, axis)


@pytest.mark.parametrize("reducer", [ak.min, ak.max, ak.argmin, ak.argmax])
def test_option_over_list_without_fill(reducer):
    check(LAYOUTS["option-over-list"], reducer, -1)


@pytest.mark.parametrize("forget_length", [False, True])
@pytest.mark.parametrize("name", ["var", "indexed-option", "option-over-list"])
def test_typetracer(name, forget_length):
    layout = LAYOUTS[name]
    typetracer = ak.Array(layout.to_typetracer(forget_length=forget_length))
    packed = ak.to_packed(ak.Array(layout))
    for reducer in [ak.sum, ak.argmax]:
        assert (
            reducer(typetracer, axis=1).layout.form
            == reducer(packed, axis=1).layout.form
        )
