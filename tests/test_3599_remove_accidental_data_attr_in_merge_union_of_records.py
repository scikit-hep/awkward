# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def virtualize(array):
    form, length, container = ak.to_buffers(array)
    new_container = {k: lambda v=v: v for k, v in container.items()}
    return ak.from_buffers(form, length, new_container)


def test_merge_union_of_records():
    a1 = ak.Array([{"a": 1, "b": 2}])
    a2 = ak.Array([{"b": 3.3, "c": 4.4}])
    c = ak.concatenate((a1, a2))
    c = virtualize(c)

    assert c.tolist() == [{"a": 1, "b": 2}, {"b": 3.3, "c": 4.4}]

    assert str(c.type) == "2 * union[{a: int64, b: int64}, {b: float64, c: float64}]"

    d = ak.merge_union_of_records(c)

    assert d.tolist() == [{"a": 1, "b": 2, "c": None}, {"a": None, "b": 3.3, "c": 4.4}]

    assert str(d.type) == "2 * {a: ?int64, b: float64, c: ?float64}"


def test_merge_union_of_records_2():
    a1 = ak.Array([{"a": 1, "b": 2}])
    a2 = ak.Array([{"b": 3.3, "c": 4.4}, {"b": None, "c": None}])
    c = ak.concatenate((a1, a2))
    c = virtualize(c)

    assert c.tolist() == [
        {"a": 1, "b": 2},
        {"b": 3.3, "c": 4.4},
        {"b": None, "c": None},
    ]

    assert str(c.type) == "3 * union[{a: int64, b: int64}, {b: ?float64, c: ?float64}]"

    d = ak.merge_union_of_records(c)

    assert d.tolist() == [
        {"a": 1, "b": 2, "c": None},
        {"a": None, "b": 3.3, "c": 4.4},
        {"a": None, "b": None, "c": None},
    ]

    assert str(d.type) == "3 * {a: ?int64, b: ?float64, c: ?float64}"


def test_merge_union_of_records_3():
    a1 = ak.Array([[[[{"a": 1, "b": 2}]]]])
    a2 = ak.Array([[[[{"b": 3.3, "c": 4.4}]]]])
    c = ak.concatenate((a1, a2), axis=-1)
    c = virtualize(c)

    assert c.tolist() == [[[[{"a": 1, "b": 2}, {"b": 3.3, "c": 4.4}]]]]

    assert (
        str(c.type)
        == "1 * var * var * var * union[{a: int64, b: int64}, {b: float64, c: float64}]"
    )

    d = ak.merge_union_of_records(c, axis=-1)

    assert d.tolist() == [
        [[[{"a": 1, "b": 2, "c": None}, {"a": None, "b": 3.3, "c": 4.4}]]]
    ]

    assert str(d.type) == "1 * var * var * var * {a: ?int64, b: float64, c: ?float64}"


def test_merge_option_of_records():
    a = ak.Array([None, {"a": 1, "b": 2}])
    a = virtualize(a)

    assert str(a.type) == "2 * ?{a: int64, b: int64}"

    b = ak.merge_option_of_records(a)

    assert b.tolist() == [{"a": None, "b": None}, {"a": 1, "b": 2}]

    assert str(b.type) == "2 * {a: ?int64, b: ?int64}"


def test_merge_option_of_records_2():
    a = ak.Array([None, {"a": 1, "b": 2}, {"a": None, "b": None}])
    a = virtualize(a)

    assert str(a.type) == "3 * ?{a: ?int64, b: ?int64}"

    b = ak.merge_option_of_records(a)

    assert b.tolist() == [
        {"a": None, "b": None},
        {"a": 1, "b": 2},
        {"a": None, "b": None},
    ]

    assert str(b.type) == "3 * {a: ?int64, b: ?int64}"


def test_merge_option_of_records_3():
    a = ak.Array([[[[None, {"a": 1, "b": 2}]]]])
    a = virtualize(a)

    assert str(a.type) == "1 * var * var * var * ?{a: int64, b: int64}"

    b = ak.merge_option_of_records(a, axis=-1)

    assert b.tolist() == [[[[{"a": None, "b": None}, {"a": 1, "b": 2}]]]]

    assert str(b.type) == "1 * var * var * var * {a: ?int64, b: ?int64}"


def test_indexed():
    x = ak.to_layout([{"a": 1, "b": 2}])
    y = ak.contents.IndexedArray(
        ak.index.Index64([1]), ak.to_layout([{"c": 13, "b": 15}, {"c": 3, "b": 5}])
    )

    z = ak.concatenate((x, y))
    z = virtualize(z)

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
    z = virtualize(z)

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
    z = virtualize(z)

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
