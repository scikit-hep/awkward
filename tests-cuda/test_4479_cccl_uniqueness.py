# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

xp = pytest.importorskip("cupy")
pytest.importorskip("cuda.compute")
BACKEND = "cuda"
host = xp.asnumpy


def call(name, arrays, scalars=()):
    backend = ak.to_backend(ak.Array([1]), BACKEND).layout.backend
    kernel = backend[(name, *(x.dtype.type for x in arrays))]
    arguments = [*arrays]
    for position, value in scalars:
        arguments.insert(position, value)
    backend.maybe_kernel_error(kernel(*arguments))


@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float32,
        np.float64,
    ],
)
@pytest.mark.parametrize(
    "rows",
    [
        [],
        [[1, 1, 1]],
        [[1, 1, 2], [2, 2, 2], [1]],
        [[0, 0, 1, 1], [1, 0, 0]],
        [[1] * 513, [1] * 513],
    ],
)
def test_unique_ranges(dtype, rows):
    rows = [np.asarray(row, dtype=dtype).tolist() for row in rows]
    values = [x for row in rows for x in row]
    offsets = np.array([0, *np.cumsum([len(row) for row in rows])], dtype=np.int64)
    expected_rows = [
        [x for i, x in enumerate(row) if i == 0 or x != row[i - 1]] for row in rows
    ]
    expected = [x for row in expected_rows for x in row]
    output = xp.asarray(np.array([*values, 1, 0], dtype=dtype))
    original = host(output).copy()
    outoffsets = xp.full(len(rows) + 3, -99, dtype=np.int64)
    name = (
        "awkward_unique_ranges_bool" if dtype == np.bool_ else "awkward_unique_ranges"
    )
    call(name, [output, xp.asarray(offsets), outoffsets], [(2, len(offsets))])
    np.testing.assert_array_equal(
        host(output)[: len(expected)], np.asarray(expected, dtype=dtype)
    )
    np.testing.assert_array_equal(
        host(output)[len(expected) :], original[len(expected) :]
    )
    assert host(outoffsets).tolist() == [
        0,
        *np.cumsum([len(row) for row in expected_rows]),
        -99,
        -99,
    ]


@pytest.mark.parametrize("dtype", [np.int64, np.float64, np.bool_])
def test_empty_range_legacy_order(dtype):
    values = [3, 2, 1, 0] if dtype != np.bool_ else [1, 1, 1, 0]
    output = xp.asarray(values, dtype=dtype)
    offsets = xp.asarray([0, 0, 3, 3], dtype=np.int64)
    outoffsets = xp.full(6, -99, dtype=np.int64)
    name = (
        "awkward_unique_ranges_bool" if dtype == np.bool_ else "awkward_unique_ranges"
    )
    call(name, [output, offsets, outoffsets], [(2, 4)])
    expected = [1, 1, 0, 0] if dtype == np.bool_ else [3, 3, 1, 0]
    assert host(output).tolist() == expected
    assert host(outoffsets).tolist() == (
        [0, 1, 2, 3, -99, -99] if dtype == np.bool_ else [0, 1, 3, 4, -99, -99]
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_equality(dtype):
    values = xp.asarray([0.0, -0.0, 1.0, np.nan, np.nan, 99.0], dtype=dtype)
    offsets = xp.asarray([0, 5], dtype=np.int64)
    outoffsets = xp.full(4, -99, dtype=np.int64)
    call("awkward_unique_ranges", [values, offsets, outoffsets], [(2, 2)])
    result = host(values)
    np.testing.assert_array_equal(result[:4], [0.0, 1.0, np.nan, np.nan])
    assert not np.signbit(result[0])
    assert host(outoffsets).tolist() == [0, 4, -99, -99]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_first_nan_legacy_order(dtype):
    # Extra readable/writable capacity makes the legacy first-NaN behavior defined.
    values = xp.asarray([np.nan, 1.0, 77.0, 88.0], dtype=dtype)
    offsets = xp.asarray([0, 2], dtype=np.int64)
    outoffsets = xp.full(3, -99, dtype=np.int64)
    call("awkward_unique_ranges", [values, offsets, outoffsets], [(2, 2)])
    np.testing.assert_array_equal(host(values), [np.nan, np.nan, np.nan, 88.0])
    assert host(outoffsets).tolist() == [0, 3, -99]


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize(
    "source,starts,expected",
    [
        ([0], [], [0]),
        ([0, 2, 4], [0, 0, 3, 3, 6], [0, 0, 2, 2, 4, 4]),
        ([0, 2, 2, 3, 4, 5], [0, 2, 4, 6], [0, 2, 2, 3, 5, 5]),
    ],
)
def test_unique_offsets(dtype, source, starts, expected):
    output = xp.full(len(expected) + 2, -99, dtype=dtype)
    call(
        "awkward_unique_offsets",
        [
            output,
            xp.asarray(source, dtype=np.int64),
            xp.asarray(starts, dtype=np.int64),
        ],
        [(1, len(source)), (4, len(starts))],
    )
    assert host(output).tolist() == [*expected, -99, -99]


@pytest.mark.parametrize(
    "offsets,nulls,index,expected_offsets",
    [
        ([0], [0], [0], [0]),
        ([0, 0, 0], [1, 1, 0], [-1, -1], [0, 1, 2]),
        ([0, 2, 2, 3], [0, 0, 1, 0, 1, 0], [0, 1, -1, 2, -1], [0, 3, 3, 5]),
        ([0, 1, 3], [0, 0, 0, 0], [0, 1, 2], [0, 1, 3]),
    ],
)
def test_index_and_offsets(offsets, nulls, index, expected_offsets):
    output = xp.full(len(index) + 2, -99, dtype=np.int64)
    outoffsets = xp.full(len(offsets) + 2, -99, dtype=np.int64)
    call(
        "awkward_IndexedArray_unique_next_index_and_offsets_64",
        [
            output,
            outoffsets,
            xp.asarray(offsets, dtype=np.int64),
            xp.asarray(nulls, dtype=np.int64),
        ],
        [(4, len(offsets) - 1)],
    )
    assert host(output).tolist() == [*index, -99, -99]
    assert host(outoffsets).tolist() == [*expected_offsets, -99, -99]


@pytest.mark.parametrize(
    "words",
    [
        [],
        [b"", b"", b""],
        [b"a", b"a", b"b", b"b", b"c"],
        [b"a\x00b", b"a\x00b", b"a\x00c", b"\x80", b"\xff"],
        [b"x"] * 513,
        [b"x" * 512, b"x" * 512, b"x" * 512 + b"a"],
        [b"a", b"b", b"a"],
    ],
)
def test_unique_strings(words):
    data = b"".join(words)
    output = xp.asarray(np.frombuffer(data + b"tail", dtype=np.uint8)).copy()
    original = host(output).copy()
    offsets = xp.asarray([0, *np.cumsum([len(x) for x in words])], dtype=np.int64)
    outoffsets = xp.full(len(words) + 3, -99, dtype=np.int64)
    length = xp.full(2, -99, dtype=np.int64)
    call(
        "awkward_NumpyArray_unique_strings",
        [output, offsets, outoffsets, length],
        [(2, len(offsets))],
    )
    expected_words = [
        word for i, word in enumerate(words) if i == 0 or word != words[i - 1]
    ]
    expected = b"".join(expected_words)
    assert bytes(host(output)[: len(expected)]) == expected
    np.testing.assert_array_equal(
        host(output)[len(expected) :], original[len(expected) :]
    )
    assert host(length).tolist() == [len(expected_words) + 1, -99]
    expected_offsets = [0, *np.cumsum([len(x) for x in expected_words])]
    assert host(outoffsets).tolist() == [
        *expected_offsets,
        *([-99] * (len(words) + 3 - len(expected_offsets))),
    ]


@pytest.mark.parametrize(
    "values,axis",
    [
        ([[3, 1, 3], [], [2, 2], [1]], -1),
        ([[True, True, False], [], [False]], -1),
        (["a\x00c", "a\x00b", "a\x00b", "", "é", "é"], -1),
        ([b"a", b"a", b"\xff", b""], -1),
        ([[3, 1, 3], [], [2, 2], [1]], None),
    ],
)
def test_layout_unique(values, axis):
    source = ak.Array(values)
    expected = ak.Array(ak._do.unique(source.layout, axis=axis)).to_list()
    array = ak.to_backend(source, BACKEND)
    actual = ak.Array(ak._do.unique(array.layout, axis=axis)).to_list()
    assert actual == expected
    assert ak._do.is_unique(array.layout, axis=axis) == ak._do.is_unique(
        source.layout, axis=axis
    )


@pytest.mark.parametrize(
    "dtype,values,expected",
    [
        (np.int64, [-(2**63), -(2**63), 2**63 - 1, 2**63 - 1], [-(2**63), 2**63 - 1]),
        (np.uint64, [2**64 - 1, 2**64 - 1, 2**63, 2**63, 0], [2**64 - 1, 2**63, 0]),
    ],
)
def test_full_width_integer_equality(dtype, values, expected):
    output = xp.asarray(values, dtype=dtype)
    offsets = xp.asarray([0, len(values)], dtype=np.int64)
    outoffsets = xp.full(2, -99, dtype=np.int64)
    call("awkward_unique_ranges", [output, offsets, outoffsets], [(2, 2)])
    assert host(output)[: len(expected)].tolist() == expected
    assert host(outoffsets).tolist() == [0, len(expected)]
