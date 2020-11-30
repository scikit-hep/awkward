# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward1 as ak  # noqa: F401


# https://github.com/scikit-hep/awkward-1.0/issues/459#issuecomment-694941328
#
# So the rules would be,
#    * if arrays have different `__array__` or `__record__` parameters, they are not equal;
#    * if they otherwise have different parameters, the types can be equal, but merging
#      (concatenation, option-simplify, or union-simplify) removes parameters other than
#      `__array__` and `__record__`.


def test_0459_types():
    plain_plain = ak.Array([0.0, 1.1, 2.2, 3.3, 4.4])
    array_plain = ak.with_parameter(plain_plain, "__array__", "zoinks")
    plain_isdoc = ak.with_parameter(plain_plain, "__doc__", "This is a zoink.")
    array_isdoc = ak.with_parameter(array_plain, "__doc__", "This is a zoink.")
    assert ak.parameters(plain_plain) == {}
    assert ak.parameters(array_plain) == {"__array__": "zoinks"}
    assert ak.parameters(plain_isdoc) == {"__doc__": "This is a zoink."}
    assert ak.parameters(array_isdoc) == {
        "__array__": "zoinks",
        "__doc__": "This is a zoink.",
    }

    assert ak.type(plain_plain) == ak.type(plain_plain)
    assert ak.type(array_plain) == ak.type(array_plain)
    assert ak.type(plain_isdoc) == ak.type(plain_isdoc)
    assert ak.type(array_isdoc) == ak.type(array_isdoc)

    assert ak.type(plain_plain) != ak.type(array_plain)
    assert ak.type(array_plain) != ak.type(plain_plain)

    assert ak.type(plain_plain) == ak.type(plain_isdoc)
    assert ak.type(plain_isdoc) == ak.type(plain_plain)

    assert ak.type(array_plain) == ak.type(array_isdoc)
    assert ak.type(array_isdoc) == ak.type(array_plain)

    assert ak.type(plain_isdoc) != ak.type(array_isdoc)
    assert ak.type(array_isdoc) != ak.type(plain_isdoc)


def test_0459():
    plain_plain = ak.Array([0.0, 1.1, 2.2, 3.3, 4.4])
    array_plain = ak.with_parameter(plain_plain, "__array__", "zoinks")
    plain_isdoc = ak.with_parameter(plain_plain, "__doc__", "This is a zoink.")
    array_isdoc = ak.with_parameter(array_plain, "__doc__", "This is a zoink.")
    assert ak.parameters(plain_plain) == {}
    assert ak.parameters(array_plain) == {"__array__": "zoinks"}
    assert ak.parameters(plain_isdoc) == {"__doc__": "This is a zoink."}
    assert ak.parameters(array_isdoc) == {
        "__array__": "zoinks",
        "__doc__": "This is a zoink.",
    }

    assert ak.parameters(ak.concatenate([plain_plain, plain_plain])) == {}
    assert ak.parameters(ak.concatenate([array_plain, array_plain])) == {
        "__array__": "zoinks"
    }
    assert ak.parameters(ak.concatenate([plain_isdoc, plain_isdoc])) == {
        "__doc__": "This is a zoink."
    }
    assert ak.parameters(ak.concatenate([array_isdoc, array_isdoc])) == {
        "__array__": "zoinks",
        "__doc__": "This is a zoink.",
    }

    assert isinstance(
        ak.concatenate([plain_plain, plain_plain]).layout, ak.layout.NumpyArray
    )
    assert isinstance(
        ak.concatenate([array_plain, array_plain]).layout, ak.layout.NumpyArray
    )
    assert isinstance(
        ak.concatenate([plain_isdoc, plain_isdoc]).layout, ak.layout.NumpyArray
    )
    assert isinstance(
        ak.concatenate([array_isdoc, array_isdoc]).layout, ak.layout.NumpyArray
    )

    assert ak.parameters(ak.concatenate([plain_plain, array_plain])) == {}
    assert ak.parameters(ak.concatenate([plain_isdoc, array_isdoc])) == {}
    assert ak.parameters(ak.concatenate([array_plain, plain_plain])) == {}
    assert ak.parameters(ak.concatenate([array_isdoc, plain_isdoc])) == {}

    assert isinstance(
        ak.concatenate([plain_plain, array_plain]).layout, ak.layout.UnionArray8_64
    )
    assert isinstance(
        ak.concatenate([plain_isdoc, array_isdoc]).layout, ak.layout.UnionArray8_64
    )
    assert isinstance(
        ak.concatenate([array_plain, plain_plain]).layout, ak.layout.UnionArray8_64
    )
    assert isinstance(
        ak.concatenate([array_isdoc, plain_isdoc]).layout, ak.layout.UnionArray8_64
    )

    assert ak.parameters(ak.concatenate([plain_plain, plain_isdoc])) == {}
    assert ak.parameters(ak.concatenate([array_plain, array_isdoc])) == {
        "__array__": "zoinks"
    }
    assert ak.parameters(ak.concatenate([plain_isdoc, plain_plain])) == {}
    assert ak.parameters(ak.concatenate([array_isdoc, array_plain])) == {
        "__array__": "zoinks"
    }

    assert isinstance(
        ak.concatenate([plain_plain, plain_isdoc]).layout, ak.layout.NumpyArray
    )
    assert isinstance(
        ak.concatenate([array_plain, array_isdoc]).layout, ak.layout.NumpyArray
    )
    assert isinstance(
        ak.concatenate([plain_isdoc, plain_plain]).layout, ak.layout.NumpyArray
    )
    assert isinstance(
        ak.concatenate([array_isdoc, array_plain]).layout, ak.layout.NumpyArray
    )


def test_0522():
    content1 = ak.Array([0.0, 1.1, 2.2, 3.3, 4.4]).layout
    content2 = ak.Array([0, 100, 200, 300, 400]).layout
    tags = ak.layout.Index8(np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1], np.int8))
    index = ak.layout.Index64(np.array([0, 1, 2, 0, 1, 3, 4, 2, 3, 4], np.int64))
    unionarray = ak.Array(ak.layout.UnionArray8_64(tags, index, [content1, content2]))
    assert unionarray.tolist() == [0.0, 1.1, 2.2, 0, 100, 3.3, 4.4, 200, 300, 400]

    assert (unionarray + 10).tolist() == [
        10.0,
        11.1,
        12.2,
        10,
        110,
        13.3,
        14.4,
        210,
        310,
        410,
    ]
    assert (10 + unionarray).tolist() == [
        10.0,
        11.1,
        12.2,
        10,
        110,
        13.3,
        14.4,
        210,
        310,
        410,
    ]

    assert (unionarray + range(0, 100, 10)).tolist() == [
        0.0,
        11.1,
        22.2,
        30,
        140,
        53.3,
        64.4,
        270,
        380,
        490,
    ]
    assert (range(0, 100, 10) + unionarray).tolist() == [
        0.0,
        11.1,
        22.2,
        30,
        140,
        53.3,
        64.4,
        270,
        380,
        490,
    ]

    assert (unionarray + np.arange(0, 100, 10)).tolist() == [
        0.0,
        11.1,
        22.2,
        30,
        140,
        53.3,
        64.4,
        270,
        380,
        490,
    ]
    assert (np.arange(0, 100, 10) + unionarray).tolist() == [
        0.0,
        11.1,
        22.2,
        30,
        140,
        53.3,
        64.4,
        270,
        380,
        490,
    ]

    assert (unionarray + ak.Array(np.arange(0, 100, 10))).tolist() == [
        0.0,
        11.1,
        22.2,
        30,
        140,
        53.3,
        64.4,
        270,
        380,
        490,
    ]
    assert (ak.Array(np.arange(0, 100, 10)) + unionarray).tolist() == [
        0.0,
        11.1,
        22.2,
        30,
        140,
        53.3,
        64.4,
        270,
        380,
        490,
    ]

    assert (unionarray + unionarray).tolist() == [
        0.0,
        2.2,
        4.4,
        0,
        200,
        6.6,
        8.8,
        400,
        600,
        800,
    ]
