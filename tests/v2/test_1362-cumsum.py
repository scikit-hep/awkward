# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401
import numpy as np  # noqa: F401

layout = ak._v2.contents.NumpyArray(np.array([1, 50, 100, 200, 500, 1000]))


def test_numpy():
    assert layout.cumsum().to_list() == [1, 51, 151, 351, 851, 1851]


def test_listoffset():
    b = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 2, 4, 6])), layout
    )

    assert b.cumsum().to_list() == [[1, 51], [100, 300], [500, 1500]]
    assert b.cumsum(axis=0).to_list() == [[1, 50], [101, 250], [601, 1250]]


def test_regular():
    c = ak._v2.contents.RegularArray(layout, 2, 2)
    assert c.cumsum().tolist() == [[1, 51], [100, 300], [500, 1500]]
    assert c.cumsum(axis=0).to_list() == [[1, 50], [101, 250], [601, 1250]]


def test_list():
    d = ak._v2.contents.ListArray(
        ak._v2.index.Index64(np.array([0, 2, 4])),
        ak._v2.index.Index64(np.array([2, 4, 6])),
        layout,
    )
    assert d.cumsum().tolist() == [[1, 51], [100, 300], [500, 1500]]
    assert d.cumsum(axis=0).to_list() == [[1, 50], [101, 250], [601, 1250]]


def test_indexed():
    e = ak._v2.contents.IndexedArray(
        ak._v2.index.Index64(np.array([0, 2, 1, 3, 4, 5])), layout
    )

    assert e.cumsum().tolist() == [1, 101, 151, 351, 851, 1851]


def test_indexedoption():
    e = ak._v2.contents.IndexedOptionArray(
        ak._v2.index.Index64(np.array([0, -1, 2, 3, -1, 5])), layout
    )

    assert e.cumsum().tolist() == [1, None, 101, 301, None, 1301]


def test_listoffset_indexed():
    # List of option
    g = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 2, 4, 6])),
        ak._v2.contents.IndexedOptionArray(
            ak._v2.index.Index64(np.array([0, -1, 2, 3, 4, 5])), layout
        ),
    )

    assert g.cumsum(axis=0).tolist() == [[1, None], [101, 200], [601, 1200]]
    assert g.cumsum(axis=1).tolist() == [[1, None], [100, 300], [500, 1500]]


def test_indexed_listoffset():
    # Option of list
    h = ak._v2.contents.IndexedOptionArray(
        ak._v2.index.Index64(np.array([0, -1, 2])),
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(np.array([0, 2, 4, 6])), layout
        ),
    )
    assert h.cumsum(axis=0).tolist() == [[1, 50], None, [501, 1050]]
    assert h.cumsum(axis=1).tolist() == [[1, 51], None, [500, 1500]]


def test_listoffset_listoffset_indexedoption():
    # List of list option
    i = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 3])),
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(np.array([0, 2, 4, 6])),
            ak._v2.contents.IndexedOptionArray(
                ak._v2.index.Index64(np.array([0, -1, 2, 3, 4, 5])), layout
            ),
        ),
    )
    assert i.cumsum(axis=0).tolist() == [[[1, None], [100, 200], [500, 1000]]]
    assert i.cumsum(axis=1).tolist() == [[[1, None], [101, 200], [601, 1200]]]
    assert i.cumsum(axis=2).tolist() == [[[1, None], [100, 300], [500, 1500]]]
