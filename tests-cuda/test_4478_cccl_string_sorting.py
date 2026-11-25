# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

xp = pytest.importorskip("cupy")
pytest.importorskip("cuda.compute")
BACKEND = "cuda"
host = xp.asnumpy


def buffers(words, prefix=b"", suffix=b""):
    offsets = np.concatenate(
        ([len(prefix)], len(prefix) + np.cumsum([len(x) for x in words]))
    )
    return xp.asarray(
        np.frombuffer(prefix + b"".join(words) + suffix, dtype=np.uint8)
    ), xp.asarray(offsets, dtype=np.int64)


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("stable", [True, False])
@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize(
    "words,parents",
    [
        ([], []),
        ([b"", b"", b""], [0, 0, 0]),
        ([b"a\x00c", b"a\x00b", b"a", b"", b"\xff", b"\x80"], [5, 5, 5, 5, 5, 5]),
        ([b"z", b"a", b"b", b"a", b"z", b"a"], [4, 4, 7, 7, 4, 4]),
        ([b"same"] * 513, [0] * 513),
        ([b"x" * 512 + b"b", b"x" * 512 + b"a", b"x" * 512], [0, 0, 0]),
    ],
)
def test_string_argsort(ascending, stable, local, words, parents):
    data, offsets = buffers(words, prefix=b"unused", suffix=b"tail")
    out = xp.full(len(words) + 2, -123, dtype=np.int64)
    parents_array = xp.asarray(parents, dtype=np.int64)
    backend = ak.to_backend(ak.Array([1]), BACKEND).layout.backend
    kernel = backend[
        "awkward_ListOffsetArray_argsort_strings",
        np.int64,
        np.int64,
        np.uint8,
        np.int64,
        np.int64,
    ]
    backend.maybe_kernel_error(
        kernel(
            out,
            parents_array,
            len(words),
            data,
            offsets[:-1],
            offsets[1:],
            stable,
            ascending,
            local,
        )
    )
    result = host(out).tolist()
    assert result[len(words) :] == [-123, -123]
    start = 0
    while start < len(words):
        stop = start + 1
        while stop < len(words) and parents[stop] == parents[start]:
            stop += 1
        order = result[start:stop]
        absolute = [x + start if local else x for x in order]
        assert sorted(absolute) == list(range(start, stop))
        assert [words[x] for x in absolute] == sorted(
            words[start:stop], reverse=not ascending
        )
        if stable:
            assert absolute == sorted(
                range(start, stop), key=words.__getitem__, reverse=not ascending
            )
        start = stop


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("stable", [True, False])
@pytest.mark.parametrize(
    "words",
    [
        [],
        [b""],
        [b"", b"", b""],
        [b"a\x00c", b"a\x00b", b"a", b"", b"\xff", b"\x80", b"a\x00b"],
        [b"same"] * 513,
        [b"x" * 512 + b"b", b"x" * 512 + b"a", b"x" * 512],
    ],
)
def test_packed_string_sort(ascending, stable, words):
    data, offsets = buffers(words, prefix=b"unused", suffix=b"tail")
    size = sum(map(len, words))
    output = xp.full(size + 3, 77, dtype=np.uint8)
    outoffsets = xp.full(len(words) + 3, -123, dtype=np.int64)
    backend = ak.to_backend(ak.Array([1]), BACKEND).layout.backend
    kernel = backend[
        "awkward_NumpyArray_sort_asstrings_uint8",
        np.uint8,
        np.uint8,
        np.int64,
        np.int64,
    ]
    backend.maybe_kernel_error(
        kernel(output, data, offsets, len(offsets), outoffsets, ascending, stable)
    )
    expected = sorted(words, reverse=not ascending)
    assert bytes(host(output)[:size]) == b"".join(expected)
    assert host(output)[size:].tolist() == [77] * 3
    assert host(outoffsets)[: len(words) + 1].tolist() == [
        0,
        *np.cumsum([len(x) for x in expected]),
    ]
    assert host(outoffsets)[len(words) + 1 :].tolist() == [-123, -123]


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("stable", [True, False])
@pytest.mark.parametrize(
    "values",
    [
        ["z", "a", "", "é", "😀", "a\x00c", "a\x00b", "a"],
        [b"z", b"a", b"", b"\xff", b"\x80", b"a\x00c", b"a\x00b", b"a"],
    ],
)
def test_highlevel_string_sort(ascending, stable, values):
    rows = [values, [], values[::-1], values[:1]]
    array = ak.to_backend(ak.Array(rows), BACKEND)
    expected = [sorted(row, reverse=not ascending) for row in rows]
    assert ak.sort(array, ascending=ascending, stable=stable).to_list() == expected
    indices = ak.argsort(array, ascending=ascending, stable=stable)
    assert array[indices].to_list() == expected
    if stable:
        assert indices.to_list() == [
            sorted(range(len(row)), key=row.__getitem__, reverse=not ascending)
            for row in rows
        ]
