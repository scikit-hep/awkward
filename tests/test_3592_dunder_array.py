# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest
from packaging.version import parse as parse_version

import awkward as ak


@pytest.mark.skipif(
    parse_version(np.__version__) < parse_version("2.0.0"),
    reason="NumPy 2 is required for the copy kwarg in np.asarray",
)
@pytest.mark.parametrize(
    "iterable",
    (
        [1.1, 2.2, 3.3, 4.4, 5.5],
        [[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]],
        ["one", "two", "three"],
        [b"one", b"two", b"three"],
    ),
)
def test_numpy2(iterable):
    nparray = np.asarray(iterable)
    akarray = ak.highlevel.Array(nparray, check_valid=True)
    out = np.asarray(akarray)
    assert np.array_equal(out, nparray)

    if np.issubdtype(nparray.dtype, np.floating):
        # copy=None and dtype=None, no copy is required
        out = np.asarray(akarray, dtype=None, copy=None)
        assert np.array_equal(out, nparray)
        assert np.shares_memory(akarray.layout.data, nparray)
        assert np.shares_memory(out, nparray)

        # copy=None and the same dtype is requested, no copy is required
        out = np.asarray(akarray, dtype=np.float64, copy=None)
        assert np.array_equal(out, nparray)
        assert np.shares_memory(akarray.layout.data, nparray)
        assert np.shares_memory(out, nparray)

        # changing dtype and copy=None, copy is required
        out = np.asarray(akarray, dtype=np.float32, copy=None)
        assert np.array_equal(out, nparray.astype(np.float32))
        assert np.shares_memory(akarray.layout.data, nparray)
        assert not np.shares_memory(out, nparray)

        # copy=True, copy is enforced
        out = np.asarray(akarray, dtype=None, copy=True)
        assert np.array_equal(out, nparray)
        assert np.shares_memory(akarray.layout.data, nparray)
        assert not np.shares_memory(out, nparray)

        # copy=True and the same dtype is requested, copy is enforced
        out = np.asarray(akarray, dtype=np.float64, copy=True)
        assert np.array_equal(out, nparray)
        assert np.shares_memory(akarray.layout.data, nparray)
        assert not np.shares_memory(out, nparray)

        # copy=True and different dtype is requested, copy is enforced
        out = np.asarray(akarray, dtype=np.float32, copy=True)
        assert np.array_equal(out, nparray.astype(np.float32))
        assert np.shares_memory(akarray.layout.data, nparray)
        assert not np.shares_memory(out, nparray)

        # copy=False and dtype=None, no copy is required
        out = np.asarray(akarray, dtype=None, copy=False)
        assert np.array_equal(out, nparray)
        assert np.shares_memory(akarray.layout.data, nparray)
        assert np.shares_memory(out, nparray)

        # copy=False and the same dtype is requested, no copy is required
        out = np.asarray(akarray, dtype=np.float64, copy=False)
        assert np.array_equal(out, nparray)
        assert np.shares_memory(akarray.layout.data, nparray)
        assert np.shares_memory(out, nparray)

        # copy=False and a different dtype is requested, copy is required
        with pytest.raises(ValueError):
            np.asarray(akarray, dtype=np.float32, copy=False)

        # copy=None and dtype=None, no copy is required
        out = np.array(akarray, dtype=None, copy=None)
        assert np.array_equal(out, nparray)
        assert np.shares_memory(akarray.layout.data, nparray)
        assert np.shares_memory(out, nparray)

        # copy=None and the same dtype is requested, no copy is required
        out = np.array(akarray, dtype=np.float64, copy=None)
        assert np.array_equal(out, nparray)
        assert np.shares_memory(akarray.layout.data, nparray)
        assert np.shares_memory(out, nparray)

        # changing dtype and copy=None, copy is required
        out = np.array(akarray, dtype=np.float32, copy=None)
        assert np.array_equal(out, nparray.astype(np.float32))
        assert np.shares_memory(akarray.layout.data, nparray)
        assert not np.shares_memory(out, nparray)

        # copy=True, copy is enforced
        out = np.array(akarray, dtype=None, copy=True)
        assert np.array_equal(out, nparray)
        assert np.shares_memory(akarray.layout.data, nparray)
        assert not np.shares_memory(out, nparray)

        # copy=True and the same dtype is requested, copy is enforced
        out = np.array(akarray, dtype=np.float64, copy=True)
        assert np.array_equal(out, nparray)
        assert np.shares_memory(akarray.layout.data, nparray)
        assert not np.shares_memory(out, nparray)

        # copy=True and different dtype is requested, copy is enforced
        out = np.array(akarray, dtype=np.float32, copy=True)
        assert np.array_equal(out, nparray.astype(np.float32))
        assert np.shares_memory(akarray.layout.data, nparray)
        assert not np.shares_memory(out, nparray)

        # copy=False and dtype=None, no copy is required
        out = np.array(akarray, dtype=None, copy=False)
        assert np.array_equal(out, nparray)
        assert np.shares_memory(akarray.layout.data, nparray)
        assert np.shares_memory(out, nparray)

        # copy=False and the same dtype is requested, no copy is required
        out = np.array(akarray, dtype=np.float64, copy=False)
        assert np.array_equal(out, nparray)
        assert np.shares_memory(akarray.layout.data, nparray)
        assert np.shares_memory(out, nparray)

        # copy=False and a different dtype is requested, copy is required
        with pytest.raises(ValueError):
            np.array(akarray, dtype=np.float32, copy=False)


@pytest.mark.skipif(
    parse_version(np.__version__) >= parse_version("2.0.0"),
    reason="NumPy 1 does not support the copy kwarg in np.asarray but does in np.array",
)
@pytest.mark.parametrize(
    "iterable",
    (
        [1.1, 2.2, 3.3, 4.4, 5.5],
        [[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]],
        ["one", "two", "three"],
        [b"one", b"two", b"three"],
    ),
)
def test_numpy1(iterable):
    nparray = np.asarray(iterable)
    akarray = ak.highlevel.Array(nparray, check_valid=True)
    out = np.asarray(akarray)
    assert np.array_equal(out, nparray)

    if np.issubdtype(nparray.dtype, np.floating):
        # asarray with dtype=None, no copy is required
        out = np.asarray(akarray, dtype=None)
        assert np.array_equal(out, nparray)
        assert np.shares_memory(akarray.layout.data, nparray)
        assert np.shares_memory(out, nparray)

        # asarray with same dtype, no copy is required
        out = np.asarray(akarray, dtype=np.float64)
        assert np.array_equal(out, nparray)
        assert np.shares_memory(akarray.layout.data, nparray)
        assert np.shares_memory(out, nparray)

        # asarray with different dtype, copy is required
        out = np.asarray(akarray, dtype=np.float32)
        assert np.array_equal(out, nparray.astype(np.float32))
        assert np.shares_memory(akarray.layout.data, nparray)
        assert not np.shares_memory(out, nparray)

        # copy=True and dtype=None, copy is enforced
        out = np.array(akarray, dtype=None, copy=True)
        assert np.array_equal(out, nparray)
        assert np.shares_memory(akarray.layout.data, nparray)
        assert not np.shares_memory(out, nparray)

        # copy=True and the same dtype is requested, copy is enforced
        out = np.array(akarray, dtype=np.float64, copy=True)
        assert np.array_equal(out, nparray)
        assert np.shares_memory(akarray.layout.data, nparray)
        assert not np.shares_memory(out, nparray)

        # copy=True and different dtype is requested, copy is enforced
        out = np.array(akarray, dtype=np.float32, copy=True)
        assert np.array_equal(out, nparray.astype(np.float32))
        assert np.shares_memory(akarray.layout.data, nparray)
        assert not np.shares_memory(out, nparray)

        # copy=False and dtype=None, no copy is required
        out = np.array(akarray, dtype=None, copy=False)
        assert np.array_equal(out, nparray)
        assert np.shares_memory(akarray.layout.data, nparray)
        assert np.shares_memory(out, nparray)

        # copy=False and the same dtype is requested, no copy is required
        out = np.array(akarray, dtype=np.float64, copy=False)
        assert np.array_equal(out, nparray)
        assert np.shares_memory(akarray.layout.data, nparray)
        assert np.shares_memory(out, nparray)

        # copy=False and a different dtype is requested, copy is required
        out = np.array(akarray, dtype=np.float32, copy=False)
        assert np.array_equal(out, nparray.astype(np.float32))
        assert np.shares_memory(akarray.layout.data, nparray)
        assert not np.shares_memory(out, nparray)
