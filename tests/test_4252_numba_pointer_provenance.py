# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import gc
import os
import subprocess
import sys

import numpy as np
import pytest

import awkward as ak

numba = pytest.importorskip("numba")
ak.numba.register_and_check()


@pytest.mark.parametrize("values", [[1, 2, 3], [True, False, True], [1.25, 2.5, 3.75]])
def test_typed_pointer_loads(values):
    @numba.njit
    def read(array, index):
        return array[index]

    array = ak.Array(values)
    for i, expected in enumerate(values):
        assert read(array, i) == expected
    llvm = read.inspect_llvm(read.signatures[0])
    # Test the generated IR, not just numerical results: LLVM optimizations can
    # expose provenance bugs only for particular loops and CPU cost models.
    assert "inttoptr" not in llvm
    assert "ptrtoint" not in llvm


@pytest.mark.parametrize("readonly", [False, True])
def test_metadata_and_pointer_storage(readonly):
    values = np.arange(6, dtype=np.int64)
    values.flags.writeable = not readonly
    array = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 2, 6])), ak.contents.NumpyArray(values)
        )
    )
    array.numba_type  # noqa: B018
    lookup = array._numbaview.lookup
    assert lookup.bufferptrs.dtype == np.dtype(np.uint8)
    for position, metadata in zip(lookup.positions, lookup.arrayptrs, strict=True):
        assert metadata == (position if isinstance(position, int) else 0)

    @numba.njit
    def read(array):
        return array[1][2]

    assert read(array) == 4
    assert "inttoptr" not in read.inspect_llvm(read.signatures[0])


@pytest.mark.parametrize(
    "dtype", [np.bool_, np.int8, np.int64, np.float64, np.complex128]
)
def test_sliced_asarray_and_lifetime(dtype):
    @numba.njit
    def asarray(array):
        return np.asarray(array[-1][1:])

    @numba.njit
    def sliced(array):
        return array[1:]

    array = ak.Array(np.arange(12).astype(dtype).reshape(3, 4))
    view = sliced(array)
    expected = ak.to_numpy(array[-1][1:]).copy()
    del array
    gc.collect()
    np.testing.assert_array_equal(asarray(view), expected)
    assert "inttoptr" not in asarray.inspect_llvm(asarray.signatures[0])


def test_constant_pointer_table_is_live():
    values = np.array([1, 2, 3], dtype=np.int64)
    array = ak.Array(values)

    @numba.njit
    def read(index):
        return array[index]

    assert read(1) == 2
    values[1] = 99
    gc.collect()
    assert read(1) == 99


def test_nested_records_masks_and_strings():
    array = ak.Array([{"x": ["é", "a\x00b"]}, {"x": []}, {"x": ["🌍", None]}])

    @numba.njit
    def read(array):
        return array[0].x[1], array[2].x[0], array[2].x[1]

    assert read(array) == ("a\x00b", "🌍", None)


@pytest.mark.skipif(sys.platform == "emscripten", reason="requires fresh subprocesses")
def test_disk_cache_with_global_and_argument_arrays(tmp_path):
    script = tmp_path / "cached_arrays.py"
    script.write_text(
        """
import sys
import warnings

import awkward as ak
import numba
from numba.core.errors import NumbaWarning

ak.numba.register_and_check()
value, expected_hits = map(int, sys.argv[1:])
array = ak.Array([value, value + 1])

@numba.njit(cache=True)
def global_read(index):
    return array[index]

@numba.njit(cache=True)
def argument_read(array, index):
    return array[index]

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", NumbaWarning)
    assert global_read(1) == value + 1
assert any(
    issubclass(w.category, NumbaWarning)
    and "dynamic globals" in str(w.message)
    for w in caught
), caught

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", NumbaWarning)
    assert argument_read(array, 1) == value + 1
assert not any(issubclass(w.category, NumbaWarning) for w in caught), caught
assert sum(argument_read.stats.cache_hits.values()) == expected_hits
assert sum(argument_read.stats.cache_misses.values()) == 1 - expected_hits
"""
    )
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "cache")}
    # Both processes use the same source and disk cache but different buffers
    # and values. Global pointers must be rebuilt; argument code must be reused.
    for value, hits in [(17, 0), (29, 1)]:
        result = subprocess.run(
            [sys.executable, str(script), str(value), str(hits)],
            check=False,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
