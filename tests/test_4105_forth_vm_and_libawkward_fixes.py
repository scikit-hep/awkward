# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import sys

import numpy as np
import pytest

import awkward as ak

forth_only = pytest.mark.skipif(
    sys.byteorder == "big",
    reason="AwkwardForth not yet supported on big-endian systems",
)


def test_from_json_nullable_record_missing_option_key():
    # A switch fall-through in nulls_for_optiontype left the instruction stack
    # unbalanced, raising a spurious "JSON schema mismatch" instead of null-filling.
    schema = {
        "type": "array",
        "items": {
            "type": ["object", "null"],
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": ["integer", "null"]},
            },
        },
    }
    out = ak.from_json('[{"x": 1}, null, {"x": 2, "y": 3}]', schema=schema)
    assert out.tolist() == [{"x": 1, "y": None}, None, {"x": 2, "y": 3}]


def test_arraybuilder_clear_then_reuse():
    # RecordBuilder::clear() used to desync the contents_/keys_ parallel arrays
    # and leave length_ == -1, breaking __len__ and reuse after clear().
    builder = ak.ArrayBuilder()
    with builder.record():
        builder.field("x").integer(1)
        builder.field("y").real(2.5)

    ext = builder._layout
    assert len(ext) == 1

    ext.clear()
    assert len(ext) == 0
    assert "RecordArray" in ext.form()

    with builder.record():
        builder.field("x").integer(7)
        builder.field("y").real(8.5)
    assert len(ext) == 1
    assert builder.snapshot().tolist() == [{"x": 7, "y": 8.5}]


@forth_only
def test_forthmachine_output_initial_size_zero():
    # maybe_resize() looped forever when output_initial_size == 0 (reservation
    # stayed 0 under geometric growth).
    vm = ak.forth.ForthMachine64(
        "output out int64 5 0 do i out <- stack loop",
        output_initial_size=0,
    )
    vm.run({})
    assert vm.output("out").tolist() == [0, 1, 2, 3, 4]


@forth_only
@pytest.mark.parametrize(
    ("nbits", "nbytes"),
    [(32, 4), (64, 8)],
)
def test_forthmachine_nbit_mask_large_widths(nbits, nbytes):
    # mask = (1 << bit_width) - 1 shifted an int literal: UB for bit_width >= 31.
    expected = (1 << nbits) - 1
    vm = ak.forth.ForthMachine64(
        f"input data output out uint64 1 data #{nbits}bit-> out",
        output_initial_size=64,
    )
    data = np.frombuffer(b"\xff" * nbytes, dtype=np.uint8)
    vm.run({"data": data})
    (result,) = vm.output("out").tolist()
    assert (result & 0xFFFFFFFFFFFFFFFF) == (expected & 0xFFFFFFFFFFFFFFFF)
