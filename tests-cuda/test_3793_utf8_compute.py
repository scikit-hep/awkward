# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from awkward._backends.cupy import CupyBackend

cp = pytest.importorskip("cupy")
pytest.importorskip("cuda.compute")

spec = importlib.util.spec_from_file_location(
    "utf8_contracts",
    Path(__file__).resolve().parents[1] / "tests/test_3793_utf8_compute.py",
)
contracts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contracts)


@pytest.mark.parametrize("strings", contracts.CASES)
@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("origin", [0, 3])
def test_conversion(strings, dtype, origin):
    data, offsets = contracts.buffers(strings, dtype, origin)
    width = max(map(len, strings), default=0) + 2
    with cp.cuda.Stream(non_blocking=True) as stream:
        data, offsets = cp.asarray(data), cp.asarray(offsets)
        maximum = cp.full(3, 99, dtype=cp.int64)
        output = cp.full(width * len(strings) + 4, 99, dtype=cp.uint32)
        backend = CupyBackend.instance()
        assert (
            backend[
                "awkward_NumpyArray_prepare_utf8_to_utf32_padded",
                np.uint8,
                dtype,
                np.int64,
            ](data, offsets, len(offsets), maximum)
            is None
        )
        assert (
            backend[
                "awkward_NumpyArray_utf8_to_utf32_padded", np.uint8, dtype, np.uint32
            ](data, offsets, len(offsets), width, output)
            is None
        )
        stream.synchronize()
    assert maximum.get().tolist() == [width - 2, 99, 99]
    assert (
        output.get().tolist()
        == [ord(c) for x in strings for c in x + "\x00" * (width - len(x))] + [99] * 4
    )


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("row", [0, 1, 2])
@pytest.mark.parametrize("invalid", [0x80, 0xBF, 0xF8, 0xFF])
def test_error(dtype, row, invalid):
    data = np.asarray([97, 98, 99, 100, 101, 102], dtype=np.uint8)
    data[row * 2 + 1] = invalid
    expected = np.full(11, 99, dtype=np.uint32)
    for previous in range(row):
        expected[previous * 3 : previous * 3 + 3] = [
            data[previous * 2],
            data[previous * 2 + 1],
            0,
        ]
    expected[row * 3] = data[row * 2]
    with cp.cuda.Stream(non_blocking=True):
        output = cp.full(11, 99, dtype=cp.uint32)
        error = CupyBackend.instance()[
            "awkward_NumpyArray_utf8_to_utf32_padded", np.uint8, dtype, np.uint32
        ](cp.asarray(data), cp.asarray([0, 2, 4, 6], dtype=dtype), 4, 3, output)
        assert (
            error.str
            == b"could not convert UTF8 code point to UTF32: invalid byte in UTF8 string"
        )
        assert error.id == 9223372036854775807
        assert error.attempt == invalid
        np.testing.assert_array_equal(output.get(), expected)
