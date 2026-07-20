# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def _argmin_pair(array, mask):
    array = ak.typetracer.length_zero_if_typetracer(array)

    assert not mask
    # Find location of minimum 0 slot
    return ak.argmin(array["0"], axis=-1, keepdims=False, mask_identity=mask)


def test_positional_record_reducer_with_shifts():
    # Missing values *inside* lists, above the records: the positional
    # result must be expressed in the original (with-Nones) coordinates,
    # i.e. the `shifts` correction must be applied to the override's
    # row-relative answer. Regression test for the offsets-pipeline
    # migration dropping `shifts` in RecordArray._reduce_next.
    behavior = {(ak.argmin, "pair"): _argmin_pair}

    x = ak.Array([[3.0, 99.0, 1.0], [98.0, 5.0], [4.0]])
    y = 2 * x
    z = ak.zip((x, y), with_name="pair", behavior=behavior)
    is_valid = ak.Array([[True, False, True], [False, True], [True]])
    z = z.mask[is_valid]

    # The None sits *before* the minimum in the first two rows, so the
    # record-reduced positions must be shifted: same answer as argmin on
    # the equivalent plain array.
    plain = ak.Array([[3.0, None, 1.0], [None, 5.0], [4.0]])
    expected = ak.argmin(plain, axis=-1, mask_identity=False)
    assert expected.to_list() == [2, 1, 0]
    assert ak.argmin(z, axis=-1, mask_identity=False).to_list() == expected.to_list()

    assert ak.argmin(z, axis=-1, mask_identity=True).to_list() == [2, 1, 0]


def test_positional_record_reducer_with_shifts_all_none_row():
    behavior = {(ak.argmin, "pair"): _argmin_pair}

    x = ak.Array([[3.0, 99.0, 1.0], [98.0, 5.0], [4.0]])
    y = 2 * x
    z = ak.zip((x, y), with_name="pair", behavior=behavior)
    is_valid = ak.Array([[True, False, True], [False, False], [True]])
    z = z.mask[is_valid]

    assert ak.argmin(z, axis=-1, mask_identity=False).to_list() == [2, -1, 0]
    assert ak.argmin(z, axis=-1, mask_identity=True).to_list() == [2, None, 0]


def test_positional_record_reducer_with_shifts_typetracer():
    # The shifts correction must also work on the typetracer backend
    # (record overrides return length-zero NumPy-backed layouts there).
    behavior = {(ak.argmin, "pair"): _argmin_pair}

    x = ak.Array([[3.0, 99.0, 1.0], [98.0, 5.0], [4.0]])
    y = 2 * x
    z = ak.zip((x, y), with_name="pair", behavior=behavior)
    is_valid = ak.Array([[True, False, True], [False, True], [True]])
    z = z.mask[is_valid]
    tt = ak.Array(z.layout.to_typetracer(forget_length=True), behavior=behavior)

    assert str(ak.argmin(tt, axis=-1, mask_identity=False).type) == "## * int64"
    assert str(ak.argmin(tt, axis=-1, mask_identity=True).type) == "## * ?int64"


def test_positional_record_reducer_without_shifts_unchanged():
    # No missing values: the override's row-relative answer passes through
    # untouched (no double `starts` adjustment).
    behavior = {(ak.argmin, "pair"): _argmin_pair}

    x = ak.Array([[3.0, 99.0, 1.0], [98.0, 5.0], [4.0]])
    y = 2 * x
    z = ak.zip((x, y), with_name="pair", behavior=behavior)

    assert ak.argmin(z, axis=-1, mask_identity=False).to_list() == [2, 1, 0]
