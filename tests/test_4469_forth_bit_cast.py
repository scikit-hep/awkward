# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import sys

import numpy as np
import pytest

import awkward as ak

pytestmark = pytest.mark.skipif(
    sys.byteorder == "big",
    reason="AwkwardForth not yet supported on big-endian systems",
)


@pytest.mark.parametrize("machine", [ak.forth.ForthMachine32, ak.forth.ForthMachine64])
@pytest.mark.parametrize("repeated", [False, True])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("byteorder", ["<", ">"])
@pytest.mark.parametrize(
    "code,dtype",
    [
        ("h", "int16"),
        ("H", "uint16"),
        ("i", "int32"),
        ("I", "uint32"),
        ("q", "int64"),
        ("Q", "uint64"),
        ("n", "intp"),
        ("N", "uintp"),
        ("f", "float32"),
        ("d", "float64"),
    ],
)
def test_endian_scalar_and_bulk_bits(machine, repeated, offset, byteorder, code, dtype):
    dtype = np.dtype(dtype)
    unsigned = np.dtype(f"uint{dtype.itemsize * 8}")
    if dtype.kind == "f":
        # Negative zero, infinity, a quiet NaN with payload, and a subnormal.
        bits = (
            [0x80000000, 0x7F800000, 0x7FC12345, 1]
            if dtype.itemsize == 4
            else [0x8000000000000000, 0x7FF0000000000000, 0x7FF8123456789ABC, 1]
        )
    else:
        bits = [0, 1, 1 << (dtype.itemsize * 8 - 1), (1 << (dtype.itemsize * 8)) - 1]
    expected = np.asarray(bits, dtype=unsigned)
    raw = bytearray(offset) + bytearray(
        expected.astype(unsigned.newbyteorder(byteorder))
    )
    data = np.frombuffer(raw, dtype=unsigned.newbyteorder(byteorder), offset=offset)
    original = data.tobytes()
    marker = "!" if byteorder == ">" else ""
    read = (
        f"4 data #{marker}{code}-> out"
        if repeated
        else f"4 0 do data {marker}{code}-> out loop"
    )
    vm = machine(f"input data output out {dtype.name} {read}")
    vm.run({"data": data})
    np.testing.assert_array_equal(vm.output("out").view(unsigned), expected)
    assert data.tobytes() == original


@pytest.mark.parametrize("code,dtype", [("n", "intp"), ("N", "uintp")])
def test_big_endian_pointer_sized_stack(code, dtype):
    values = np.asarray([1, 127, 12345], dtype=dtype)
    data = values.astype(values.dtype.newbyteorder(">"))
    vm = ak.forth.ForthMachine64(f"input data 3 data #!{code}-> stack")
    vm.run({"data": data})
    assert vm.stack == values.tolist()


@pytest.mark.parametrize("machine", [ak.forth.ForthMachine32, ak.forth.ForthMachine64])
@pytest.mark.parametrize("repeated", [False, True])
@pytest.mark.parametrize("big_endian", [False, True])
def test_endian_readonly_input_requirement(machine, repeated, big_endian):
    data = np.asarray([1, 2], dtype=">i4" if big_endian else "<i4")
    data.flags.writeable = False
    marker = "!" if big_endian else ""
    read = (
        f"2 data #{marker}i-> out" if repeated else f"2 0 do data {marker}i-> out loop"
    )
    vm = machine(f"input data output out int32 {read}")
    if repeated and big_endian:
        with pytest.raises(ValueError, match="read-only"):
            vm.run({"data": data})
    else:
        vm.run({"data": data})
        np.testing.assert_array_equal(vm.output("out"), [1, 2])
