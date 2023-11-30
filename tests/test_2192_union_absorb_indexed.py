# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test_simplified_indexed():
    numbers = ak.contents.IndexedArray(
        ak.index.Index64([0, 2, 3]),
        ak.contents.NumpyArray(np.arange(10, dtype=np.int64)),
    )
    records = ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.array([4.0, 3.0, 1.0], dtype=np.int64))], ["x"]
    )
    result = ak.concatenate((numbers, records), highlevel=False)
    assert result.is_equal_to(
        ak.contents.UnionArray(
            ak.index.Index8([0, 0, 0, 1, 1, 1]),
            ak.index.Index64([0, 2, 3, 0, 1, 2]),
            [
                ak.contents.NumpyArray(np.arange(10, dtype=np.int64)),
                ak.contents.RecordArray(
                    [ak.contents.NumpyArray(np.array([4.0, 3.0, 1.0], dtype=np.int64))],
                    ["x"],
                ),
            ],
        )
    )


def test_simplified_indexed_categorical():
    numbers = ak.contents.IndexedArray(
        ak.index.Index64([0, 2, 3]),
        ak.contents.NumpyArray(np.arange(10, dtype=np.int64)),
        parameters={"__array__": "categorical"},
    )
    records = ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.array([4.0, 3.0, 1.0], dtype=np.int64))], ["x"]
    )
    result = ak.concatenate((numbers, records), highlevel=False)
    assert result.is_equal_to(
        ak.contents.UnionArray(
            ak.index.Index8([0, 0, 0, 1, 1, 1]),
            ak.index.Index64([0, 1, 2, 0, 1, 2]),
            [
                ak.contents.IndexedArray(
                    ak.index.Index64([0, 2, 3]),
                    ak.contents.NumpyArray(np.arange(10, dtype=np.int64)),
                    parameters={"__array__": "categorical"},
                ),
                ak.contents.RecordArray(
                    [ak.contents.NumpyArray(np.array([4.0, 3.0, 1.0], dtype=np.int64))],
                    ["x"],
                ),
            ],
        )
    )


def test_merge_indexed():
    records = ak.contents.IndexedArray(
        ak.index.Index64([0, 2, 3]),
        ak.contents.RecordArray(
            [
                ak.contents.NumpyArray(
                    np.array([4.0, 3.0, 1.0, 9.0, 8.0, 7.0], dtype=np.int64)
                )
            ],
            ["x"],
            parameters={"inner": "bar", "drop": "this"},
        ),
        parameters={"outer": "foo", "ignore": "me"},
    )
    union = ak.contents.UnionArray(
        ak.index.Index8([0, 0, 0, 1, 1, 1]),
        ak.index.Index64([0, 1, 2, 0, 1, 2]),
        [
            ak.contents.NumpyArray(np.arange(10, dtype=np.int64)),
            ak.contents.RecordArray(
                [ak.contents.NumpyArray(np.array([4.0, 3.0, 1.0], dtype=np.int64))],
                ["x"],
                parameters={"outer": "foo", "inner": "bar"},
            ),
        ],
    )
    result = ak.concatenate((union, records), highlevel=False)

    assert result.is_equal_to(
        ak.contents.UnionArray(
            ak.index.Index8([0, 0, 0, 1, 1, 1, 1, 1, 1]),
            ak.index.Index64([0, 1, 2, 0, 1, 2, 3, 5, 6]),
            [
                ak.contents.NumpyArray(np.arange(10, dtype=np.int64)),
                ak.contents.RecordArray(
                    [
                        ak.contents.NumpyArray(
                            np.array(
                                [4.0, 3.0, 1.0, 4.0, 3.0, 1.0, 9.0, 8.0, 7.0],
                                dtype=np.int64,
                            )
                        )
                    ],
                    ["x"],
                ),
            ],
        )
    )
    # This test might be a bit strict; any code that views `layout.parameters` will change this result to `{}`
    assert result.contents[0]._parameters is None
    assert result.contents[1]._parameters == {"outer": "foo", "inner": "bar"}


def test_merge_indexed_categorical():
    records = ak.contents.IndexedArray(
        ak.index.Index64([0, 2, 3]),
        ak.contents.RecordArray(
            [
                ak.contents.NumpyArray(
                    np.array([4.0, 3.0, 1.0, 9.0, 8.0, 7.0], dtype=np.int64)
                )
            ],
            ["x"],
            parameters={"inner": "bar", "drop": "this"},
        ),
        parameters={"outer": "foo", "ignore": "me", "__array__": "categorical"},
    )
    union = ak.contents.UnionArray(
        ak.index.Index8([0, 0, 0, 1, 1, 1]),
        ak.index.Index64([0, 1, 2, 0, 1, 2]),
        [
            ak.contents.NumpyArray(np.arange(10, dtype=np.int64)),
            ak.contents.IndexedArray(
                ak.index.Index64([0, 1, 2]),
                ak.contents.RecordArray(
                    [ak.contents.NumpyArray(np.array([4.0, 3.0, 1.0], dtype=np.int64))],
                    ["x"],
                    parameters={"inner": "bar"},
                ),
                parameters={"outer": "foo", "__array__": "categorical"},
            ),
        ],
    )
    with pytest.raises(
        NotImplementedError,
        match=r"merging categorical arrays is currently not implemented",
    ):
        ak.concatenate((union, records), highlevel=False)


def test_merge_indexed_mixed_categorical():
    records = ak.contents.IndexedArray(
        ak.index.Index64([0, 2, 3]),
        ak.contents.RecordArray(
            [
                ak.contents.NumpyArray(
                    np.array([4.0, 3.0, 1.0, 9.0, 8.0, 7.0], dtype=np.int64)
                )
            ],
            ["x"],
            parameters={"inner": "bar", "drop": "this"},
        ),
        parameters={"outer": "foo", "ignore": "me"},
    )
    union = ak.contents.UnionArray(
        ak.index.Index8([0, 0, 0, 1, 1, 1]),
        ak.index.Index64([0, 1, 2, 0, 1, 2]),
        [
            ak.contents.NumpyArray(np.arange(10, dtype=np.int64)),
            ak.contents.IndexedArray(
                ak.index.Index64([0, 1, 2]),
                ak.contents.RecordArray(
                    [ak.contents.NumpyArray(np.array([4.0, 3.0, 1.0], dtype=np.int64))],
                    ["x"],
                    parameters={"inner": "bar"},
                ),
                parameters={"outer": "foo", "__array__": "categorical"},
            ),
        ],
    )
    result = ak.concatenate((union, records), highlevel=False)
    assert result.is_equal_to(
        ak.contents.UnionArray(
            ak.index.Index8([0, 0, 0, 1, 1, 1, 1, 1, 1]),
            ak.index.Index64([0, 1, 2, 0, 1, 2, 3, 5, 6]),
            [
                ak.contents.NumpyArray(np.arange(10, dtype=np.int64)),
                ak.contents.RecordArray(
                    [
                        ak.contents.NumpyArray(
                            np.array(
                                [4.0, 3.0, 1.0, 4.0, 3.0, 1.0, 9.0, 8.0, 7.0],
                                dtype=np.int64,
                            )
                        )
                    ],
                    ["x"],
                ),
            ],
        ),
    )
    # This test might be a bit strict; any code that views `layout.parameters` will change this result to `{}`
    assert result.contents[0]._parameters is None
    assert result.contents[1].parameters == {"outer": "foo", "inner": "bar"}
