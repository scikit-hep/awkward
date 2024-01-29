# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from itertools import product

import pytest

import awkward as ak

behavior_1 = {"foo": "bar"}
behavior_2 = {"baz": "bargh!"}
behavior = {**behavior_1, **behavior_2}


@pytest.mark.parametrize(
    ("func", "axis"),
    [
        pytest.param(ak.softmax, 0, marks=pytest.mark.xfail()),
        (ak.std, 0),
        (ak.var, 0),
        (ak.softmax, 1),
        (ak.std, 1),
        (ak.var, 1),
        *product(
            (
                ak.any,
                ak.min,
                ak.argmin,
                ak.sum,
                ak.ptp,
                ak.count_nonzero,
                lambda *args, **kwargs: ak.moment(*args, **kwargs, n=3),
                ak.argmax,
                ak.all,
                ak.mean,
                ak.max,
                ak.prod,
                ak.count,
            ),
            ([0, 1]),
        ),
    ],
)
def test_reducers(axis, func):
    assert isinstance(
        func([[1, 2, 3, 4], [5], [10]], axis=axis, highlevel=True), ak.Array
    )
    assert isinstance(
        func([[1, 2, 3, 4], [5], [10]], axis=axis, highlevel=False), ak.contents.Content
    )
    assert (
        func(
            ak.Array([[1, 2, 3, 4], [5], [10]], behavior=behavior_1),
            axis=axis,
            highlevel=True,
            behavior=behavior_2,
        ).behavior
        == behavior_2
    )
    assert (
        func(
            ak.Array([[1, 2, 3, 4], [5], [10]], behavior=behavior_1),
            axis=axis,
            highlevel=True,
        ).behavior
        == behavior_1
    )


@pytest.mark.parametrize("func", [ak.covar, ak.corr, ak.linear_fit])
def test_covar(func):
    assert isinstance(
        func(
            [[1, 2, 3, 4], [5], [10]],
            [[4, 4, 0, 2], [1], [10]],
            axis=-1,
            highlevel=True,
        ),
        ak.Array,
    )
    assert isinstance(
        func(
            [[1, 2, 3, 4], [5], [10]],
            [[4, 4, 0, 2], [1], [10]],
            axis=-1,
            highlevel=False,
        ),
        ak.contents.Content,
    )
    assert (
        func(
            ak.Array(
                [[1, 2, 3, 4], [5], [10]],
                behavior=behavior_1,
            ),
            [[4, 4, 0, 2], [1], [10]],
            axis=-1,
            highlevel=True,
            behavior=behavior_2,
        ).behavior
        == behavior_2
    )
    assert (
        func(
            [[1, 2, 3, 4], [5], [10]],
            ak.Array(
                [[4, 4, 0, 2], [1], [10]],
                behavior=behavior_1,
            ),
            axis=-1,
            highlevel=True,
            behavior=behavior_2,
        ).behavior
        == behavior_2
    )
    assert (
        func(
            ak.Array(
                [[1, 2, 3, 4], [5], [10]],
                behavior=behavior_1,
            ),
            [[4, 4, 0, 2], [1], [10]],
            axis=-1,
            highlevel=True,
        ).behavior
        == behavior_1
    )
    assert (
        func(
            [[1, 2, 3, 4], [5], [10]],
            ak.Array(
                [[4, 4, 0, 2], [1], [10]],
                behavior=behavior_1,
            ),
            axis=-1,
            highlevel=True,
        ).behavior
        == behavior_1
    )
    assert (
        func(
            [[1, 2, 3, 4], [5], [10]],
            [[4, 4, 0, 2], [1], [10]],
            weight=ak.Array(
                [[1, 2, 3, 2], [1], [1]],
                behavior=behavior_1,
            ),
            axis=-1,
            highlevel=True,
            behavior=behavior_2,
        ).behavior
        == behavior_2
    )
    assert (
        func(
            [[1, 2, 3, 4], [5], [10]],
            [[4, 4, 0, 2], [1], [10]],
            weight=ak.Array(
                [[1, 2, 3, 2], [1], [1]],
                behavior=behavior_1,
            ),
            axis=-1,
            highlevel=True,
        ).behavior
        == behavior_1
    )
