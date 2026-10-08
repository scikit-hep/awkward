# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import sys

import numpy as np
import pytest

import awkward as ak

pytestmark = pytest.mark.skipif(
    sys.byteorder == "big",
    reason="AwkwardForth not yet supported on big-endian systems",
)


@pytest.mark.parametrize(
    "dtype",
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    ],
)
def test_output_strides(dtype):
    vm = ak.forth.ForthMachine32(f"output out {dtype} 4 0 do i out <- stack loop")
    vm.run({})
    out = vm.output("out")
    assert out.dtype == np.dtype(dtype)
    assert out.strides == (out.itemsize,)
    assert out.tolist() == np.arange(4).astype(dtype).tolist()


def test_word_bytecodes_strides():
    vm = ak.forth.ForthMachine32(": foo 1 2 + ; foo")
    word = vm["foo"]
    offsets = vm.bytecodes_offsets
    index = vm.dictionary.index("foo") + 1
    assert len(word) > 1
    assert word.strides == (word.itemsize,)
    assert word.tolist() == vm.bytecodes[offsets[index] : offsets[index + 1]]
