# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401
import numpy as np  # noqa: F401


def test_fix_longlong_type_passed_to_index_1530():
    a = np.asarray([0, 5, 10, 15, 20, 25], dtype=np.longlong)
    index = ak._v2.index.Index64(a)
    content = ak._v2.contents.NumpyArray(
        np.asarray(
            [101, 118, 116, 45, 48, 101, 118, 116, 45, 49, 101, 118], dtype=np.int64
        )
    )
    listoffsetarray = ak._v2.contents.ListOffsetArray(index, content)

    assert ak._v2.to_list(listoffsetarray) == [
        [101, 118, 116, 45, 48],
        [101, 118, 116, 45, 49],
        [101, 118],
        [],
        [],
    ]


def test_typetracer_view_method():
    a = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.longlong)
    b = ak._v2.contents.NumpyArray(a)
    c = np.array([7, 3, 3, 5], dtype=np.longlong)
    e = ak._v2.index.Index(b.typetracer[c])
    f = ak._v2.index.Index(a)

    assert e.form == f.form
    assert ak._v2.to_list(b[c]) == ak._v2.to_list(a[c])
    assert b.typetracer[c].form == b[c].form
