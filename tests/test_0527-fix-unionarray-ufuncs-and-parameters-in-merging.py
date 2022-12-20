# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

# https://github.com/scikit-hep/awkward-1.0/issues/459#issuecomment-694941328
#
# So the rules would be,
#    * if arrays have different `__array__` or `__record__` parameters, they are not equal;
#    * if they otherwise have different parameters, the types can be equal, but merging
#      (concatenation, option-simplify, or union-simplify) removes parameters other than
#      `__array__` and `__record__`.


def test_0459_types():

    plain_plain = ak.highlevel.Array([0.0, 1.1, 2.2, 3.3, 4.4])
    array_plain = ak.operations.with_parameter(plain_plain, "__array__", "zoinks")
    plain_isdoc = ak.operations.with_parameter(
        plain_plain, "__doc__", "This is a zoink."
    )
    array_isdoc = ak.operations.with_parameter(
        array_plain, "__doc__", "This is a zoink."
    )

    assert ak.operations.parameters(plain_plain) == {}
    assert ak.operations.parameters(array_plain) == {"__array__": "zoinks"}
    assert ak.operations.parameters(plain_isdoc) == {"__doc__": "This is a zoink."}
    assert ak.operations.parameters(array_isdoc) == {
        "__array__": "zoinks",
        "__doc__": "This is a zoink.",
    }

    assert ak.operations.type(plain_plain) == ak.operations.type(plain_plain)
    assert ak.operations.type(array_plain) == ak.operations.type(array_plain)
    assert ak.operations.type(plain_isdoc) == ak.operations.type(plain_isdoc)
    assert ak.operations.type(array_isdoc) == ak.operations.type(array_isdoc)

    assert ak.operations.type(plain_plain) != ak.operations.type(array_plain)
    assert ak.operations.type(array_plain) != ak.operations.type(plain_plain)

    assert ak.operations.type(plain_plain) == ak.operations.type(plain_isdoc)
    assert ak.operations.type(plain_isdoc) == ak.operations.type(plain_plain)

    assert ak.operations.type(array_plain) == ak.operations.type(array_isdoc)
    assert ak.operations.type(array_isdoc) == ak.operations.type(array_plain)

    assert ak.operations.type(plain_isdoc) != ak.operations.type(array_isdoc)
    assert ak.operations.type(array_isdoc) != ak.operations.type(plain_isdoc)

    assert array_plain.layout.parameters == {"__array__": "zoinks"}
    assert ak.operations.without_parameters(array_plain).layout.parameters == {}
    assert plain_isdoc.layout.parameters == {"__doc__": "This is a zoink."}
    assert ak.operations.without_parameters(plain_isdoc).layout.parameters == {}
    assert array_isdoc.layout.parameters == {
        "__array__": "zoinks",
        "__doc__": "This is a zoink.",
    }
    assert ak.operations.without_parameters(array_isdoc).layout.parameters == {}


def test_0459():
    plain_plain = ak.highlevel.Array([0.0, 1.1, 2.2, 3.3, 4.4])
    array_plain = ak.operations.with_parameter(plain_plain, "__array__", "zoinks")
    plain_isdoc = ak.operations.with_parameter(
        plain_plain, "__doc__", "This is a zoink."
    )
    array_isdoc = ak.operations.with_parameter(
        array_plain, "__doc__", "This is a zoink."
    )

    assert ak.operations.parameters(plain_plain) == {}
    assert ak.operations.parameters(array_plain) == {"__array__": "zoinks"}
    assert ak.operations.parameters(plain_isdoc) == {"__doc__": "This is a zoink."}
    assert ak.operations.parameters(array_isdoc) == {
        "__array__": "zoinks",
        "__doc__": "This is a zoink.",
    }

    assert (
        ak.operations.parameters(ak.operations.concatenate([plain_plain, plain_plain]))
        == {}
    )
    assert ak.operations.parameters(
        ak.operations.concatenate([array_plain, array_plain])
    ) == {"__array__": "zoinks"}
    assert ak.operations.parameters(
        ak.operations.concatenate([plain_isdoc, plain_isdoc])
    ) == {"__doc__": "This is a zoink."}
    assert ak.operations.parameters(
        ak.operations.concatenate([array_isdoc, array_isdoc])
    ) == {
        "__array__": "zoinks",
        "__doc__": "This is a zoink.",
    }

    assert isinstance(
        ak.operations.concatenate([plain_plain, plain_plain]).layout,
        ak.contents.NumpyArray,
    )
    assert isinstance(
        ak.operations.concatenate([array_plain, array_plain]).layout,
        ak.contents.NumpyArray,
    )
    assert isinstance(
        ak.operations.concatenate([plain_isdoc, plain_isdoc]).layout,
        ak.contents.NumpyArray,
    )
    assert isinstance(
        ak.operations.concatenate([array_isdoc, array_isdoc]).layout,
        ak.contents.NumpyArray,
    )

    assert (
        ak.operations.parameters(ak.operations.concatenate([plain_plain, array_plain]))
        == {}
    )
    assert (
        ak.operations.parameters(ak.operations.concatenate([plain_isdoc, array_isdoc]))
        == {}
    )
    assert (
        ak.operations.parameters(ak.operations.concatenate([array_plain, plain_plain]))
        == {}
    )
    assert (
        ak.operations.parameters(ak.operations.concatenate([array_isdoc, plain_isdoc]))
        == {}
    )

    assert isinstance(
        ak.operations.concatenate([plain_plain, array_plain]).layout,
        ak.contents.UnionArray,
    )
    assert isinstance(
        ak.operations.concatenate([plain_isdoc, array_isdoc]).layout,
        ak.contents.UnionArray,
    )
    assert isinstance(
        ak.operations.concatenate([array_plain, plain_plain]).layout,
        ak.contents.UnionArray,
    )
    assert isinstance(
        ak.operations.concatenate([array_isdoc, plain_isdoc]).layout,
        ak.contents.UnionArray,
    )

    assert (
        ak.operations.parameters(ak.operations.concatenate([plain_plain, plain_isdoc]))
        == {}
    )
    assert ak.operations.parameters(
        ak.operations.concatenate([array_plain, array_isdoc])
    ) == {"__array__": "zoinks"}
    assert (
        ak.operations.parameters(ak.operations.concatenate([plain_isdoc, plain_plain]))
        == {}
    )
    assert ak.operations.parameters(
        ak.operations.concatenate([array_isdoc, array_plain])
    ) == {"__array__": "zoinks"}

    assert isinstance(
        ak.operations.concatenate([plain_plain, plain_isdoc]).layout,
        ak.contents.NumpyArray,
    )
    assert isinstance(
        ak.operations.concatenate([array_plain, array_isdoc]).layout,
        ak.contents.NumpyArray,
    )
    assert isinstance(
        ak.operations.concatenate([plain_isdoc, plain_plain]).layout,
        ak.contents.NumpyArray,
    )
    assert isinstance(
        ak.operations.concatenate([array_isdoc, array_plain]).layout,
        ak.contents.NumpyArray,
    )


def test_0522():
    content1 = ak.highlevel.Array([0.0, 1.1, 2.2, 3.3, 4.4]).layout
    content2 = ak.highlevel.Array([[0], [100], [200], [300], [400]]).layout
    tags = ak.index.Index8(np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1], np.int8))
    index = ak.index.Index64(np.array([0, 1, 2, 0, 1, 3, 4, 2, 3, 4], np.int64))
    unionarray = ak.highlevel.Array(
        ak.contents.UnionArray(tags, index, [content1, content2])
    )
    assert unionarray.to_list() == [
        0.0,
        1.1,
        2.2,
        [0],
        [100],
        3.3,
        4.4,
        [200],
        [300],
        [400],
    ]

    assert (unionarray + 10).to_list() == [
        10.0,
        11.1,
        12.2,
        [10],
        [110],
        13.3,
        14.4,
        [210],
        [310],
        [410],
    ]
    assert (10 + unionarray).to_list() == [
        10.0,
        11.1,
        12.2,
        [10],
        [110],
        13.3,
        14.4,
        [210],
        [310],
        [410],
    ]

    assert (unionarray + range(0, 100, 10)).to_list() == [
        0.0,
        11.1,
        22.2,
        [30],
        [140],
        53.3,
        64.4,
        [270],
        [380],
        [490],
    ]
    assert (range(0, 100, 10) + unionarray).to_list() == [
        0.0,
        11.1,
        22.2,
        [30],
        [140],
        53.3,
        64.4,
        [270],
        [380],
        [490],
    ]

    assert (unionarray + np.arange(0, 100, 10)).tolist() == [
        0.0,
        11.1,
        22.2,
        [30],
        [140],
        53.3,
        64.4,
        [270],
        [380],
        [490],
    ]
    assert (np.arange(0, 100, 10) + unionarray).tolist() == [
        0.0,
        11.1,
        22.2,
        [30],
        [140],
        53.3,
        64.4,
        [270],
        [380],
        [490],
    ]

    assert (unionarray + ak.highlevel.Array(np.arange(0, 100, 10))).tolist() == [
        0.0,
        11.1,
        22.2,
        [30],
        [140],
        53.3,
        64.4,
        [270],
        [380],
        [490],
    ]
    assert (ak.highlevel.Array(np.arange(0, 100, 10)) + unionarray).tolist() == [
        0.0,
        11.1,
        22.2,
        [30],
        [140],
        53.3,
        64.4,
        [270],
        [380],
        [490],
    ]

    assert (unionarray + unionarray).to_list() == [
        0.0,
        2.2,
        4.4,
        [0],
        [200],
        6.6,
        8.8,
        [400],
        [600],
        [800],
    ]
