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
    # A nullable record (type ["object", "null"]) with a missing option-type
    # key used to raise a spurious "JSON schema mismatch" because the
    # instruction stack was left unbalanced (switch fall-through in
    # nulls_for_optiontype). It should null-fill the missing key, exactly like
    # the non-nullable record does.
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


def test_from_json_nonnullable_record_missing_option_key():
    # The non-nullable counterpart already worked; keep it as a guard.
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": ["integer", "null"]},
            },
        },
    }
    out = ak.from_json('[{"x": 1}, {"x": 2, "y": 3}]', schema=schema)
    assert out.tolist() == [{"x": 1, "y": None}, {"x": 2, "y": 3}]


def test_arraybuilder_clear_then_reuse():
    # RecordBuilder::clear() used to clear keys_/pointers_ while keeping
    # contents_, leaving the parallel arrays out of sync (out-of-bounds reads)
    # and length_ == -1 (ValueError from __len__). clear() must leave a
    # consistent, empty-but-structured state that can be reused.
    builder = ak.ArrayBuilder()
    with builder.record():
        builder.field("x").integer(1)
        builder.field("y").real(2.5)

    ext = builder._layout
    assert len(ext) == 1

    ext.clear()
    # No ValueError, length resets cleanly, form stays intact.
    assert len(ext) == 0
    form_after_clear = ext.form()
    assert "RecordArray" in form_after_clear

    # Reuse after clear must succeed and produce a consistent array.
    with builder.record():
        builder.field("x").integer(7)
        builder.field("y").real(8.5)
    assert len(ext) == 1
    assert builder.snapshot().tolist() == [{"x": 7, "y": 8.5}]


@forth_only
def test_forthmachine_output_initial_size_zero():
    # maybe_resize() used to loop forever when output_initial_size == 0
    # (reservation stayed 0). It must grow and run to completion.
    vm = ak.forth.ForthMachine64(
        "output out int64 5 0 do i out <- stack loop",
        output_initial_size=0,
    )
    vm.run({})
    assert vm.output("out").tolist() == [0, 1, 2, 3, 4]


@forth_only
def test_forthmachine_output_resize_factor_one():
    # A resize factor of exactly 1.0 also failed to grow geometrically.
    vm = ak.forth.ForthMachine64(
        "output out int64 5 0 do i out <- stack loop",
        output_initial_size=1,
        output_resize_factor=1.0,
    )
    vm.run({})
    assert vm.output("out").tolist() == [0, 1, 2, 3, 4]


@forth_only
@pytest.mark.parametrize(
    ("nbits", "nbytes"),
    [(32, 4), (33, 5), (40, 5), (64, 8)],
)
def test_forthmachine_nbit_mask_large_widths(nbits, nbytes):
    # mask = (1 << bit_width) - 1 shifted an int literal -> UB/wrong for
    # bit_width >= 31. Reading all-ones bytes must yield exactly nbits set.
    expected = (1 << nbits) - 1
    vm = ak.forth.ForthMachine64(
        f"input data output out uint64 1 data #{nbits}bit-> out",
        output_initial_size=64,
    )
    data = np.frombuffer(b"\xff" * nbytes, dtype=np.uint8)
    vm.run({"data": data})
    (result,) = vm.output("out").tolist()
    assert (result & 0xFFFFFFFFFFFFFFFF) == (expected & 0xFFFFFFFFFFFFFFFF)
