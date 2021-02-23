# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_partition_methods():
    one, two, three = ak.Array([1, 2, 3]), ak.Array([4, 5]), ak.Array([6])
    no = one.layout
    yes = ak.partitioned([one, two, three]).layout

    ONE, TWO, THREE = ak.Array([[1, 2], [3]]), ak.Array([[4, 5]]), ak.Array([[6]])
    NO = ONE.layout
    YES = ak.partitioned([ONE, TWO, THREE]).layout

    assert no.withparameter("hey", "you").parameter("hey") == "you"
    assert yes.withparameter("hey", "you").parameter("hey") == "you"
    assert no.type({}) == yes.type({})
    assert no.form == yes.form
    assert len(no) == 3
    assert len(yes) == 6
    assert list(no) == [1, 2, 3]
    assert list(yes) == [1, 2, 3, 4, 5, 6]
    assert no.kernels == yes.kernels == "cpu"
    assert no.caches == yes.caches
    assert no.tojson() == "[1,2,3]"
    assert yes.tojson() == "[1,2,3,4,5,6]"
    assert no.nbytes < yes.nbytes
    assert no.deep_copy().tojson() == "[1,2,3]"
    assert yes.deep_copy().tojson() == "[1,2,3,4,5,6]"
    assert no.numfields == yes.numfields
    assert no.haskey("doesn't") == yes.haskey("doesn't")
    assert no.keys() == yes.keys()
    assert no.purelist_isregular == yes.purelist_isregular
    assert no.purelist_depth == yes.purelist_depth
    assert no.branch_depth == yes.branch_depth
    assert no.minmax_depth == yes.minmax_depth
    assert list(no.getitem_nothing()) == list(yes.getitem_nothing())
    assert no.getitem_at_nowrap(2) == yes.getitem_at_nowrap(2)
    assert list(no.getitem_range_nowrap(1, 2)) == list(yes.getitem_range_nowrap(1, 2))
    assert no.validityerror() is None
    assert yes.validityerror() is None
    assert list(no.fillna(ak.to_layout([123]))) == [1, 2, 3]
    assert list(yes.fillna(ak.to_layout([123]))) == [1, 2, 3, 4, 5, 6]
    assert no.num(0) == 3
    assert yes.num(0) == 6
    assert list(NO.flatten(1)) == [1, 2, 3]
    assert list(YES.flatten(1)) == [1, 2, 3, 4, 5, 6]
    assert [np.asarray(x).tolist() for x in NO.offsets_and_flatten(1)] == [
        [0, 2, 3],
        [1, 2, 3],
    ]
    assert [np.asarray(x).tolist() for x in YES.offsets_and_flatten(1)] == [
        [0, 2, 3, 5, 6],
        [1, 2, 3, 4, 5, 6],
    ]
    assert ak.to_list(NO.rpad(3, 1)) == [[1, 2, None], [3, None, None]]
    assert ak.to_list(YES.rpad(3, 1)) == [
        [1, 2, None],
        [3, None, None],
        [4, 5, None],
        [6, None, None],
    ]
    assert ak.to_list(NO.rpad_and_clip(3, 1)) == [[1, 2, None], [3, None, None]]
    assert ak.to_list(YES.rpad_and_clip(3, 1)) == [
        [1, 2, None],
        [3, None, None],
        [4, 5, None],
        [6, None, None],
    ]
    assert no.mergeable(no)
    assert yes.mergeable(yes)
    assert ak.to_list(no.merge(no)) == [1, 2, 3, 1, 2, 3]
    assert ak.to_list(yes.merge(yes)) == [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
    assert ak.to_list(no.mergemany([no])) == [1, 2, 3, 1, 2, 3]
    assert ak.to_list(yes.mergemany([yes])) == [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
    assert NO.axis_wrap_if_negative(-1) == 1
    assert YES.axis_wrap_if_negative(-1) == 1
    assert no.count(0, False, False) == 3
    assert yes.count(0, False, False) == 6
    assert no.count_nonzero(0, False, False) == 3
    assert yes.count_nonzero(0, False, False) == 6
    assert no.sum(0, False, False) == 1 + 2 + 3
    assert yes.sum(0, False, False) == 1 + 2 + 3 + 4 + 5 + 6
    assert no.prod(0, False, False) == 1 * 2 * 3
    assert yes.prod(0, False, False) == 1 * 2 * 3 * 4 * 5 * 6
    assert no.any(0, False, False) is True
    assert yes.any(0, False, False) is True
    assert no.all(0, False, False) is True
    assert yes.all(0, False, False) is True
    assert no.min(0, False, False, None) == 1
    assert yes.min(0, False, False, None) == 1
    assert no.max(0, False, False, None) == 3
    assert yes.max(0, False, False, None) == 6
    assert no.argmin(0, False, False) == 0
    assert yes.argmin(0, False, False) == 0
    assert no.argmax(0, False, False) == 2
    assert yes.argmax(0, False, False) == 5
    assert ak.to_list(NO.localindex(1)) == [[0, 1], [0]]
    assert ak.to_list(YES.localindex(1)) == [[0, 1], [0], [0, 1], [0]]
    assert ak.to_list(NO.combinations(2, True, None, {}, 1)) == [
        [(1, 1), (1, 2), (2, 2)],
        [(3, 3)],
    ]
    assert ak.to_list(YES.combinations(2, True, None, {}, 1)) == [
        [(1, 1), (1, 2), (2, 2)],
        [(3, 3)],
        [(4, 4), (4, 5), (5, 5)],
        [(6, 6)],
    ]
    assert ak.to_list(no.sort(0, True, True)) == [1, 2, 3]
    assert ak.to_list(NO.sort(1, True, True)) == [[1, 2], [3]]
    assert ak.to_list(yes.sort(0, True, True)) == [1, 2, 3, 4, 5, 6]
    assert ak.to_list(YES.sort(0, True, True)) == [[1, 2], [3], [4, 5], [6]]
    assert ak.to_list(no.argsort(0, True, True)) == [0, 1, 2]
    assert ak.to_list(NO.argsort(1, True, True)) == [[0, 1], [0]]
    assert ak.to_list(yes.argsort(0, True, True)) == [0, 1, 2, 3, 4, 5]
    assert ak.to_list(YES.argsort(1, True, True)) == [[0, 1], [0], [0, 1], [0]]
    assert np.asarray(no.numbers_to_type("float64")).dtype == np.dtype("f8")
    assert np.asarray(yes.numbers_to_type("float64")).dtype == np.dtype("f8")
    assert no.is_unique() is True
    assert NO.is_unique() is True
    assert yes.is_unique() is True
    assert YES.is_unique() is True
    assert ak.to_list(no.copy_to("cpu")) == [1, 2, 3]
    assert ak.to_list(yes.copy_to("cpu")) == [1, 2, 3, 4, 5, 6]
    assert ak.to_list(
        no.carry(ak.layout.Index64(np.array([2, 1, 0], np.int64)), False)
    ) == [3, 2, 1]
    assert ak.to_list(
        yes.carry(ak.layout.Index64(np.array([5, 4, 3, 2, 1, 0], np.int64)), False)
    ) == [6, 5, 4, 3, 2, 1]
