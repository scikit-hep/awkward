# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import warnings

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.jax import Jax

jax = pytest.importorskip("jax")
ak.jax.register_and_check()


DTYPE_CASES = [
    np.array([1, 2, 3], dtype=dtype)
    for dtype in [
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
        "datetime64[ns]",
        "timedelta64[ns]",
    ]
]


@pytest.mark.parametrize("values", DTYPE_CASES)
def test_jax_nplike_byteswap_matches_numpy(values: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        nplike = Jax.instance()

    try:
        array = nplike.asarray(values)
    except (TypeError, ValueError, NotImplementedError):
        pytest.skip(f"jax does not support dtype {values.dtype}")

    swapped = nplike.byteswap(array)
    expected = values.byteswap(inplace=False)
    got = np.asarray(swapped)

    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    assert got.tobytes() == expected.tobytes()


def test_to_from_buffers_byteorder_roundtrip_jax():
    source = ak.Array([[1.0, 2.0], [], [3.0]], backend="jax")
    form, length, container = ak.to_buffers(source, byteorder=">")

    expected_offsets = ak._util.native_to_byteorder(
        np.array([0, 2, 2, 3], dtype=np.int64), ">"
    )
    expected_data = ak._util.native_to_byteorder(np.array([1.0, 2.0, 3.0]), ">")

    assert (
        np.asarray(container["node0-offsets"]).tobytes() == expected_offsets.tobytes()
    )
    assert np.asarray(container["node1-data"]).tobytes() == expected_data.tobytes()

    reconstructed = ak.from_buffers(
        form, length, container, backend="jax", byteorder=">"
    )
    assert ak.to_list(reconstructed) == ak.to_list(source)
    assert ak.backend(reconstructed) == "jax"


def test_from_buffers_accepts_big_endian_bytes_jax():
    form = ak.forms.NumpyForm("float64", form_key="node0")
    container = {"node0-data": np.array([1.0, 2.0, 3.0], dtype=">f8").tobytes()}

    out = ak.from_buffers(form, 3, container, backend="jax", byteorder=">")
    assert ak.to_list(out) == [1.0, 2.0, 3.0]
    assert ak.backend(out) == "jax"
