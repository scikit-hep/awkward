# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test():
    one = awkward1.Array([[0, 1, 2], [], [3, 4]])
    two = awkward1.Array([[100, 200], [300], [400, 500]])
    three = awkward1.Array([["a", "b"], ["c", "d"], ["e"]])

    assert awkward1.tolist(awkward1.cross([one])) == [[(0,), (1,), (2,)], [], [(3,), (4,)]]
    assert awkward1.tolist(awkward1.cross([one, two])) == [[(0, 100), (0, 200), (1, 100), (1, 200), (2, 100), (2, 200)], [], [(3, 400), (3, 500), (4, 400), (4, 500)]]
    assert awkward1.tolist(awkward1.cross([one, two, three])) == [[(0, 100, "a"), (0, 100, "b"), (0, 200, "a"), (0, 200, "b"), (1, 100, "a"), (1, 100, "b"), (1, 200, "a"), (1, 200, "b"), (2, 100, "a"), (2, 100, "b"), (2, 200, "a"), (2, 200, "b")], [], [(3, 400, "e"), (3, 500, "e"), (4, 400, "e"), (4, 500, "e")]]

    assert awkward1.tolist(awkward1.cross([one, two, three], nested=[0])) == [[[(0, 100, "a"), (0, 100, "b"), (0, 200, "a"), (0, 200, "b")], [(1, 100, "a"), (1, 100, "b"), (1, 200, "a"), (1, 200, "b")], [(2, 100, "a"), (2, 100, "b"), (2, 200, "a"), (2, 200, "b")]], [], [[(3, 400, "e"), (3, 500, "e")], [(4, 400, "e"), (4, 500, "e")]]]
    assert awkward1.tolist(awkward1.cross([one, two, three], nested=[1])) == [[[(0, 100, "a"), (0, 100, "b")], [(0, 200, "a"), (0, 200, "b")], [(1, 100, "a"), (1, 100, "b")], [(1, 200, "a"), (1, 200, "b")], [(2, 100, "a"), (2, 100, "b")], [(2, 200, "a"), (2, 200, "b")]], [], [[(3, 400, "e")], [(3, 500, "e")], [(4, 400, "e")], [(4, 500, "e")]]]
    assert awkward1.tolist(awkward1.cross([one, two, three], nested=[0, 1])) == [[[[(0, 100, "a"), (0, 100, "b")], [(0, 200, "a"), (0, 200, "b")]], [[(1, 100, "a"), (1, 100, "b")], [(1, 200, "a"), (1, 200, "b")]], [[(2, 100, "a"), (2, 100, "b")], [(2, 200, "a"), (2, 200, "b")]]], [], [[[(3, 400, "e")], [(3, 500, "e")]], [[(4, 400, "e")], [(4, 500, "e")]]]]

    assert awkward1.tolist(awkward1.cross([one, two, three], nested=[])) == awkward1.tolist(awkward1.cross([one, two, three], nested=False))
    assert awkward1.tolist(awkward1.cross([one, two, three], nested=[])) == awkward1.tolist(awkward1.cross([one, two, three], nested=None))
    assert awkward1.tolist(awkward1.cross([one, two, three], nested=[0, 1])) == awkward1.tolist(awkward1.cross([one, two, three], nested=True))
