# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


# https://github.com/scikit-hep/awkward-1.0/issues/459#issuecomment-694941328
#
# So the rules would be,
#    * if arrays have different `__array__` or `__record__` parameters, they are not equal;
#    * if they otherwise have different parameters, the types can be equal, but merging
#      (concatenation, option-simplify, or union-simplify) removes parameters other than
#      `__array__` and `__record__`.


def test_0459_types():
    plain_plain = awkward1.Array([0.0, 1.1, 2.2, 3.3, 4.4])
    array_plain = awkward1.with_parameter(plain_plain, "__array__", "zoinks")
    plain_isdoc = awkward1.with_parameter(plain_plain, "__doc__", "This is a zoink.")
    array_isdoc = awkward1.with_parameter(array_plain, "__doc__", "This is a zoink.")
    assert awkward1.parameters(plain_plain) == {}
    assert awkward1.parameters(array_plain) == {"__array__": "zoinks"}
    assert awkward1.parameters(plain_isdoc) == {"__doc__": "This is a zoink."}
    assert awkward1.parameters(array_isdoc) == {"__array__": "zoinks", "__doc__": "This is a zoink."}

    assert awkward1.type(plain_plain) == awkward1.type(plain_plain)
    assert awkward1.type(array_plain) == awkward1.type(array_plain)
    assert awkward1.type(plain_isdoc) == awkward1.type(plain_isdoc)
    assert awkward1.type(array_isdoc) == awkward1.type(array_isdoc)

    assert awkward1.type(plain_plain) != awkward1.type(array_plain)
    assert awkward1.type(array_plain) != awkward1.type(plain_plain)

    assert awkward1.type(plain_plain) == awkward1.type(plain_isdoc)
    assert awkward1.type(plain_isdoc) == awkward1.type(plain_plain)

    assert awkward1.type(array_plain) == awkward1.type(array_isdoc)
    assert awkward1.type(array_isdoc) == awkward1.type(array_plain)

    assert awkward1.type(plain_isdoc) != awkward1.type(array_isdoc)
    assert awkward1.type(array_isdoc) != awkward1.type(plain_isdoc)


def test_0459():
    plain_plain = awkward1.Array([0.0, 1.1, 2.2, 3.3, 4.4])
    array_plain = awkward1.with_parameter(plain_plain, "__array__", "zoinks")
    plain_isdoc = awkward1.with_parameter(plain_plain, "__doc__", "This is a zoink.")
    array_isdoc = awkward1.with_parameter(array_plain, "__doc__", "This is a zoink.")
    assert awkward1.parameters(plain_plain) == {}
    assert awkward1.parameters(array_plain) == {"__array__": "zoinks"}
    assert awkward1.parameters(plain_isdoc) == {"__doc__": "This is a zoink."}
    assert awkward1.parameters(array_isdoc) == {"__array__": "zoinks", "__doc__": "This is a zoink."}

    assert awkward1.parameters(awkward1.concatenate([plain_plain, plain_plain])) == {}
    assert awkward1.parameters(awkward1.concatenate([array_plain, array_plain])) == {"__array__": "zoinks"}
    assert awkward1.parameters(awkward1.concatenate([plain_isdoc, plain_isdoc])) == {"__doc__": "This is a zoink."}
    assert awkward1.parameters(awkward1.concatenate([array_isdoc, array_isdoc])) == {"__array__": "zoinks", "__doc__": "This is a zoink."}

    assert isinstance(awkward1.concatenate([plain_plain, plain_plain]).layout, awkward1.layout.NumpyArray)
    assert isinstance(awkward1.concatenate([array_plain, array_plain]).layout, awkward1.layout.NumpyArray)
    assert isinstance(awkward1.concatenate([plain_isdoc, plain_isdoc]).layout, awkward1.layout.NumpyArray)
    assert isinstance(awkward1.concatenate([array_isdoc, array_isdoc]).layout, awkward1.layout.NumpyArray)

    assert awkward1.parameters(awkward1.concatenate([plain_plain, array_plain])) == {}
    assert awkward1.parameters(awkward1.concatenate([plain_isdoc, array_isdoc])) == {}
    assert awkward1.parameters(awkward1.concatenate([array_plain, plain_plain])) == {}
    assert awkward1.parameters(awkward1.concatenate([array_isdoc, plain_isdoc])) == {}

    assert isinstance(awkward1.concatenate([plain_plain, array_plain]).layout, awkward1.layout.UnionArray8_64)
    assert isinstance(awkward1.concatenate([plain_isdoc, array_isdoc]).layout, awkward1.layout.UnionArray8_64)
    assert isinstance(awkward1.concatenate([array_plain, plain_plain]).layout, awkward1.layout.UnionArray8_64)
    assert isinstance(awkward1.concatenate([array_isdoc, plain_isdoc]).layout, awkward1.layout.UnionArray8_64)

    assert awkward1.parameters(awkward1.concatenate([plain_plain, plain_isdoc])) == {}
    assert awkward1.parameters(awkward1.concatenate([array_plain, array_isdoc])) == {"__array__": "zoinks"}
    assert awkward1.parameters(awkward1.concatenate([plain_isdoc, plain_plain])) == {}
    assert awkward1.parameters(awkward1.concatenate([array_isdoc, array_plain])) == {"__array__": "zoinks"}

    assert isinstance(awkward1.concatenate([plain_plain, plain_isdoc]).layout, awkward1.layout.NumpyArray)
    assert isinstance(awkward1.concatenate([array_plain, array_isdoc]).layout, awkward1.layout.NumpyArray)
    assert isinstance(awkward1.concatenate([plain_isdoc, plain_plain]).layout, awkward1.layout.NumpyArray)
    assert isinstance(awkward1.concatenate([array_isdoc, array_plain]).layout, awkward1.layout.NumpyArray)


def test_0522():
    content1 = awkward1.Array([0.0, 1.1, 2.2, 3.3, 4.4]).layout
    content2 = awkward1.Array([  0, 100, 200, 300, 400]).layout
    tags = awkward1.layout.Index8(numpy.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1], numpy.int8))
    index = awkward1.layout.Index64(numpy.array([0, 1, 2, 0, 1, 3, 4, 2, 3, 4], numpy.int64))
    unionarray = awkward1.Array(awkward1.layout.UnionArray8_64(tags, index, [content1, content2]))
    assert unionarray.tolist() == [0.0, 1.1, 2.2, 0, 100, 3.3, 4.4, 200, 300, 400]

    assert (unionarray + 10).tolist() == [10.0, 11.1, 12.2, 10, 110, 13.3, 14.4, 210, 310, 410]
    assert (10 + unionarray).tolist() == [10.0, 11.1, 12.2, 10, 110, 13.3, 14.4, 210, 310, 410]

    assert (unionarray + range(0, 100, 10)).tolist() == [0.0, 11.1, 22.2, 30, 140, 53.3, 64.4, 270, 380, 490]
    assert (range(0, 100, 10) + unionarray).tolist() == [0.0, 11.1, 22.2, 30, 140, 53.3, 64.4, 270, 380, 490]

    assert (unionarray + numpy.arange(0, 100, 10)).tolist() == [0.0, 11.1, 22.2, 30, 140, 53.3, 64.4, 270, 380, 490]
    assert (numpy.arange(0, 100, 10) + unionarray).tolist() == [0.0, 11.1, 22.2, 30, 140, 53.3, 64.4, 270, 380, 490]

    assert (unionarray + awkward1.Array(numpy.arange(0, 100, 10))).tolist() == [0.0, 11.1, 22.2, 30, 140, 53.3, 64.4, 270, 380, 490]
    assert (awkward1.Array(numpy.arange(0, 100, 10)) + unionarray).tolist() == [0.0, 11.1, 22.2, 30, 140, 53.3, 64.4, 270, 380, 490]

    assert (unionarray + unionarray).tolist() == [0.0, 2.2, 4.4, 0, 200, 6.6, 8.8, 400, 600, 800]
