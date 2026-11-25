# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import ast
import functools
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import awkward as ak


def load_callbacks(monkeypatch):
    source = Path(ak.__file__).parent / "_connect/cuda/_compute.py"
    names = {
        "_string_order",
        "awkward_ListOffsetArray_argsort_strings",
        "awkward_NumpyArray_sort_asstrings_uint8",
        "awkward_NumpyArray_unique_strings",
        "_unique_ranges",
        "awkward_unique_ranges",
        "awkward_unique_ranges_bool",
        "awkward_unique_offsets",
        "awkward_IndexedArray_unique_next_index_and_offsets_64",
    }
    tree = ast.parse(source.read_text())
    tree.body = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]

    def merge_sort(*, d_in_keys, d_out_keys, num_items, op):
        def compare(a, b):
            return -1 if op(a, b) else 1 if op(b, a) else 0

        d_out_keys[:num_items] = sorted(
            range(num_items), key=functools.cmp_to_key(compare)
        )

    def scan(*, d_in, d_out, op, num_items, **kwargs):
        data = d_in[:num_items]
        d_out[:num_items] = (
            np.cumsum(data) if op == "sum" else np.maximum.accumulate(data)
        )

    def transform(*, d_in, d_out, op, num_items):
        for i in range(num_items):
            value = op(np.int64(i))
            if d_out is not None:
                d_out[i] = value

    monkeypatch.setitem(
        sys.modules, "cuda.compute", SimpleNamespace(merge_sort=merge_sort)
    )
    namespace = {
        "__name__": "awkward._connect.cuda._compute",
        "np": np,
        "cp": np,
        "OpKind": SimpleNamespace(PLUS="sum", MAXIMUM="max"),
        "CountingIterator": lambda start: start,
        "DiscardIterator": lambda: None,
        "inclusive_scan": scan,
        "unary_transform": transform,
    }
    exec(compile(tree, str(source), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("ascending", [False, True])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize(
    "words", [[], [b"", b"a", b"a\x00c", b"a\x00b", b"a", b"\xff", b"z"]]
)
def test_actual_string_callbacks(monkeypatch, ascending, local, words):
    module = load_callbacks(monkeypatch)
    offsets = np.array([0, *np.cumsum([len(w) for w in words])], dtype=np.int64)
    data = np.frombuffer(b"".join(words), dtype=np.uint8)
    parents = np.array([i // 3 for i in range(len(words))], dtype=np.int64)
    output = np.full(len(words) + 2, -99, dtype=np.int64)
    module.awkward_ListOffsetArray_argsort_strings(
        output,
        parents,
        len(words),
        data,
        offsets[:-1],
        offsets[1:],
        True,
        ascending,
        local,
    )
    expected = []
    for start in range(0, len(words), 3):
        run = sorted(
            range(start, min(start + 3, len(words))),
            key=lambda i: words[i],
            reverse=not ascending,
        )
        expected.extend([i - start if local else i for i in run])
    assert output.tolist() == [*expected, -99, -99]
    result = np.full(len(data) + 2, 99, dtype=np.uint8)
    outoffsets = np.full(len(offsets) + 2, -99, dtype=np.int64)
    module.awkward_NumpyArray_sort_asstrings_uint8(
        result, data, offsets, len(offsets), outoffsets, ascending, True
    )
    ordered = sorted(words, reverse=not ascending)
    assert result[: len(data)].tobytes() == b"".join(ordered)
    assert outoffsets[: len(offsets)].tolist() == [
        0,
        *np.cumsum([len(w) for w in ordered]),
    ]
    assert outoffsets[len(offsets) :].tolist() == [-99, -99]
