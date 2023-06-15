# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401

import awkward as ak


def test_indexed():
    x = ak.to_layout([{"a": 1, "b": 2}])
    y = ak.contents.IndexedArray(
        ak.index.Index64([1]), ak.to_layout([{"c": 13, "b": 15}, {"c": 3, "b": 5}])
    )

    z = ak.concatenate((x, y))

    assert z.tolist() == [
        {"a": 1, "b": 2},
        {"c": 3, "b": 5},
    ]
    assert z.type == ak.types.ArrayType(
        ak.types.UnionType(
            [
                ak.types.RecordType(
                    [ak.types.NumpyType("int64"), ak.types.NumpyType("int64")],
                    ["a", "b"],
                ),
                ak.types.RecordType(
                    [ak.types.NumpyType("int64"), ak.types.NumpyType("int64")],
                    ["c", "b"],
                ),
            ]
        ),
        2,
    )

    w = ak.merge_union_of_records(z)
    assert w.type == ak.types.ArrayType(
        ak.types.RecordType(
            [
                ak.types.OptionType(ak.types.NumpyType("int64")),
                ak.types.NumpyType("int64"),
                ak.types.OptionType(ak.types.NumpyType("int64")),
            ],
            ["a", "b", "c"],
        ),
        2,
    )
    assert w.tolist() == [{"a": 1, "b": 2, "c": None}, {"a": None, "b": 5, "c": 3}]


def test_option():
    x = ak.to_layout([{"a": 1, "b": 2}])
    y = ak.to_layout([{"c": 3, "b": 5}, None])

    z = ak.concatenate((x, y))

    assert z.tolist() == [{"a": 1, "b": 2}, {"c": 3, "b": 5}, None]
    assert z.type == ak.types.ArrayType(
        ak.types.UnionType(
            [
                ak.types.OptionType(
                    ak.types.RecordType(
                        [ak.types.NumpyType("int64"), ak.types.NumpyType("int64")],
                        ["a", "b"],
                    )
                ),
                ak.types.OptionType(
                    ak.types.RecordType(
                        [ak.types.NumpyType("int64"), ak.types.NumpyType("int64")],
                        ["c", "b"],
                    )
                ),
            ]
        ),
        3,
    )

    w = ak.merge_union_of_records(z)
    assert w.type == ak.types.ArrayType(
        ak.types.OptionType(
            ak.types.RecordType(
                [
                    ak.types.OptionType(ak.types.NumpyType("int64")),
                    ak.types.NumpyType("int64"),
                    ak.types.OptionType(ak.types.NumpyType("int64")),
                ],
                ["a", "b", "c"],
            )
        ),
        3,
    )
    assert w.tolist() == [
        {"a": 1, "b": 2, "c": None},
        {"a": None, "b": 5, "c": 3},
        None,
    ]


def test_option_unmasked():
    x = ak.to_layout([{"a": 1, "b": 2}])
    y = ak.contents.UnmaskedArray(ak.to_layout([{"c": 3, "b": 5}]))

    z = ak.concatenate((x, y))

    assert z.tolist() == [{"a": 1, "b": 2}, {"c": 3, "b": 5}]
    assert z.type == ak.types.ArrayType(
        ak.types.UnionType(
            [
                ak.types.OptionType(
                    ak.types.RecordType(
                        [ak.types.NumpyType("int64"), ak.types.NumpyType("int64")],
                        ["a", "b"],
                    )
                ),
                ak.types.OptionType(
                    ak.types.RecordType(
                        [ak.types.NumpyType("int64"), ak.types.NumpyType("int64")],
                        ["c", "b"],
                    )
                ),
            ]
        ),
        2,
    )

    w = ak.merge_union_of_records(z)
    assert w.type == ak.types.ArrayType(
        ak.types.OptionType(
            ak.types.RecordType(
                [
                    ak.types.OptionType(ak.types.NumpyType("int64")),
                    ak.types.NumpyType("int64"),
                    ak.types.OptionType(ak.types.NumpyType("int64")),
                ],
                ["a", "b", "c"],
            )
        ),
        2,
    )
    assert w.tolist() == [{"a": 1, "b": 2, "c": None}, {"a": None, "b": 5, "c": 3}]
