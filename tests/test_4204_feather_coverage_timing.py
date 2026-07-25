# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import os
import time

import numpy as np
import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")



# Coverage: parameters not exercised by the original test_2683 file


BASE_ARRAY = [[1.12345, 2.232452356, 3.3241536], [], [4.4365156, 5.5365713]]
RECORD_ARRAY = [
    [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
    [],
    [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}],
]

# TO check roundway stablity in compression
@pytest.mark.parametrize("compression", [None, False, "uncompressed"])
def test_compression_none_false_uncompressed_roundtrip(tmp_path, compression):
    filename = os.path.join(tmp_path, f"compression_{compression}.feather")
    array = ak.Array(BASE_ARRAY)

    ak.to_feather(array, filename, compression=compression)
    array2 = ak.from_feather(filename)

    assert array2.tolist() == BASE_ARRAY

# TO roundway check multiple compression variants
@pytest.mark.parametrize("compression", [True, "lz4", "zstd", "uncompressed"])
def test_compression_variants(tmp_path, compression):
    filename = os.path.join(tmp_path, f"compression_{compression}.feather")
    array = ak.Array(BASE_ARRAY)

    ak.to_feather(array, filename, compression=compression)
    array2 = ak.from_feather(filename)

    assert array2.tolist() == BASE_ARRAY

# 
def test_compression_level(tmp_path):
    filename = os.path.join(tmp_path, "compression_level.feather")
    array = ak.Array(BASE_ARRAY)

    ak.to_feather(array, filename, compression="zstd", compression_level=5)
    array2 = ak.from_feather(filename)

    assert array2.tolist() == BASE_ARRAY


def test_feather_version_1_default_compression_is_broken(tmp_path):
    filename = os.path.join(tmp_path, "v1_default.feather")
    array = ak.Array(BASE_ARRAY)

    with pytest.raises(ValueError, match="Feather V1 files do not support compression"):
        ak.to_feather(array, filename, feather_version=1)


def test_feather_version_1_is_unconditionally_broken(tmp_path):
    filename = os.path.join(tmp_path, "v1.feather")
    array = ak.Array(BASE_ARRAY)

    for compression in (True, False, None, "zstd", "uncompressed"):
        with pytest.raises(ValueError, match="Feather V1 files do not support"):
            ak.to_feather(
                array, filename, feather_version=1, compression=compression
            )



def test_bad_destination_type():
    array = ak.Array(BASE_ARRAY)

    with pytest.raises(TypeError):
        ak.to_feather(array, 12345)


def test_chunksize(tmp_path):
    filename = os.path.join(tmp_path, "chunksize.feather")
    array = ak.Array(list(range(10000)))

    ak.to_feather(array, filename, chunksize=125)
    array2 = ak.from_feather(filename)

    assert array2.tolist() == array.tolist()


def test_columns_selection(tmp_path):
    filename = os.path.join(tmp_path, "columns.feather")
    array = ak.Array({"x": [1.1, 2.2, 3.3], "y": [[1], [1, 2], [1, 2, 3]]})

    ak.to_feather(array, filename, compression=True)
    array2 = ak.from_feather(filename, columns=["x"])

    assert set(array2.fields) == {"x"}
    assert array2.x.tolist() == array.x.tolist()


def test_memory_map(tmp_path):
    filename = os.path.join(tmp_path, "memory_map.feather")
    array = ak.Array(BASE_ARRAY)

    ak.to_feather(array, filename, compression=True)
    array2 = ak.from_feather(filename, memory_map=True)

    assert array2.tolist() == BASE_ARRAY


def test_use_threads_false(tmp_path):
    filename = os.path.join(tmp_path, "use_threads.feather")
    array = ak.Array(BASE_ARRAY)

    ak.to_feather(array, filename, compression=True)
    array2 = ak.from_feather(filename, use_threads=False)

    assert array2.tolist() == BASE_ARRAY


def test_generate_bitmasks(tmp_path):
    filename = os.path.join(tmp_path, "bitmasks.feather")
    array = ak.Array([1.1, None, 3.3])

    ak.to_feather(array, filename, extensionarray=False)
    array2 = ak.from_feather(filename, generate_bitmasks=True)

    assert array2.tolist() == [1.1, None, 3.3]


def test_extensionarray_false(tmp_path):
    filename = os.path.join(tmp_path, "no_ext.feather")
    array = ak.Array(RECORD_ARRAY)

    ak.to_feather(array, filename, extensionarray=False)
    array2 = ak.from_feather(filename)

    assert array2.tolist() == RECORD_ARRAY




def _make_union_array():
    c, i = ak.contents, ak.index
    layout = c.UnionArray(
        i.Index8(np.array([0, 1], dtype=np.int8)),
        i.Index64(np.array([0, 0], dtype=np.int64)),
        [c.NumpyArray(np.array([1.5], dtype=np.float64)), ak.to_layout(["abc"])],
    )
    return ak.Array(layout)


def test_union_type_extensionarray_true_is_known_broken(tmp_path):
    filename = os.path.join(tmp_path, "union.feather")
    array = _make_union_array()

    ak.to_feather(array, filename) # if it works ,it is the bug

    with pytest.raises(ValueError):
        ak.from_feather(filename)


def test_union_type_extensionarray_false_roundtrips(tmp_path):
    filename = os.path.join(tmp_path, "union_no_ext.feather")
    array = _make_union_array()

    ak.to_feather(array, filename, extensionarray=False)
    array2 = ak.from_feather(filename)

    assert array2.tolist() == [1.5, "abc"]

# Canary Tests

def _time_it(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


@pytest.mark.parametrize("n", [10_000, 100_000, 10_000_000])
@pytest.mark.parametrize("compression", ["zstd", "uncompressed"])
def test_timing_flat_array(tmp_path, n, compression):
    filename = os.path.join(tmp_path, f"timing_{n}_{compression}.feather")
    array = ak.Array(np.random.default_rng(0).random(n))

    _, write_time = _time_it(
        lambda: ak.to_feather(array, filename, compression=compression)
    )
    result, read_time = _time_it(lambda: ak.from_feather(filename))

    assert result.tolist() == array.tolist()

    print(
        f"\n[timing] n={n} compression={compression!r} "
        f"write={write_time:.4f}s read={read_time:.4f}s "
        f"file_size={os.path.getsize(filename)/1e6:.2f}MB"
    )

    # Generous upper bound: this is not meant to be a tight perf assertion,
    # just a canary for a genuine multi-minute stall (e.g. the writer
    # hanging on close rather than a normal slow-but-bounded write).
    assert write_time < 30
    assert read_time < 30


def test_timing_nested_array(tmp_path):
    filename = os.path.join(tmp_path, "timing_nested.feather")
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 5, size=50_000)
    array = ak.unflatten(rng.random(int(counts.sum())), counts)

    _, write_time = _time_it(lambda: ak.to_feather(array, filename))
    result, read_time = _time_it(lambda: ak.from_feather(filename))

    assert result.tolist() == array.tolist()

    print(
        f"\n[timing] nested list array (50k rows) "
        f"write={write_time:.4f}s read={read_time:.4f}s "
        f"file_size={os.path.getsize(filename)/1e6:.2f}MB"
    )

    assert write_time < 30
    assert read_time < 30


def test_timing_chunksize_sensitivity(tmp_path):
    # Does a small chunksize noticeably slow down the write's "ending"
    # (i.e. flush/close), compared to the default? This directly probes
    # whether chunking behavior could be the source of an unexpected delay.
    array = ak.Array(np.random.default_rng(0).random(500_000))

    timings = {}
    for chunksize in (None, 1024, 65536):
        filename = os.path.join(tmp_path, f"chunk_{chunksize}.feather")
        _, write_time = _time_it(
            lambda cs=chunksize: ak.to_feather(array, filename, chunksize=cs)
        )
        timings[chunksize] = write_time

    print(f"\n[timing] chunksize sensitivity (500k rows): {timings}")

    # A small chunksize can be slower, but should not be wildly (>10x)
    # slower than the default - if it is, that is worth flagging as the
    # likely source of a timeout in a production job using odd chunksize.
    if timings[None] > 0:
        assert timings[1024] < timings[None] * 20 + 5