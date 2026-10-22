# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")

N_NULLS = 20_000


def _peak_bytes(string):
    # Build the array outside the measured region: we only care about how much
    # memory the Arrow conversion itself allocates.
    array = ak.Array([string] + [None] * N_NULLS)
    tracemalloc.start()
    try:
        assert len(pyarrow.array(array)) == N_NULLS + 1
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.thread_unsafe(reason="tracemalloc's counters are process-global")
def test_masked_lists_do_not_copy_content():
    # Converting an option-type ListArray used to compact the content before
    # emptying the null'ed lists, copying the one non-null string once per null.
    # Peak memory must therefore not depend on the size of that string.
    #
    # Asserting a *ratio* rather than an absolute byte count keeps this robust
    # across platforms, whose baseline allocations differ.
    tiny = _peak_bytes("x")
    big = _peak_bytes("oof" * 2000)
    assert big < 2 * tiny


def test_arrow_output_is_unchanged():
    array = ak.Array(["oof" * 2000] + [None] * 1_000 + ["bar"])
    result = pyarrow.array(array)
    assert result.to_pylist() == array.to_list()
    assert result.null_count == 1_000


def test_mask_shorter_than_list_content():
    # A ByteMaskedArray's content may be longer than its mask, so `validbytes`
    # covers fewer entries than the ListArray has starts/stops.
    content = ak.contents.NumpyArray(np.arange(20, dtype=np.int64))
    listarray = ak.contents.ListArray(
        ak.index.Index64(np.array([0, 3, 6, 9, 12], dtype=np.int64)),
        ak.index.Index64(np.array([3, 6, 9, 12, 15], dtype=np.int64)),
        content,
    )
    layout = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([1, 0, 1], dtype=np.int8)),
        listarray,
        valid_when=True,
    )
    array = ak.Array(layout)
    assert pyarrow.array(array).to_pylist() == array.to_list()
