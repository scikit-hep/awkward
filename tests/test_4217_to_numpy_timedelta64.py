# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("unit", ["D", "s", "ms", "us", "ns"])
def test_indexed_option_timedelta64_to_numpy(unit):
    arr = ak.Array([np.timedelta64(1, unit), None, np.timedelta64(-3, unit)])
    result = ak.to_numpy(arr, allow_missing=True)

    assert isinstance(result, np.ma.MaskedArray)
    assert result.dtype == np.dtype(f"timedelta64[{unit}]")
    np.testing.assert_array_equal(result.mask, [False, True, False])
    assert result[0] == np.timedelta64(1, unit)
    assert result[2] == np.timedelta64(-3, unit)
    assert ak.array_equal(ak.from_numpy(result), arr, same_content_types=False)


def test_unitless_and_all_missing_timedelta64_to_numpy():
    all_none = ak.Array(
        ak.contents.IndexedOptionArray(
            ak.index.Index64(np.array([-1, -1], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype="timedelta64[s]")),
        )
    )
    res_all_none = ak.to_numpy(all_none, allow_missing=True)
    assert isinstance(res_all_none, np.ma.MaskedArray)
    assert res_all_none.dtype == np.dtype("timedelta64[s]")
    np.testing.assert_array_equal(res_all_none.mask, [True, True])
    assert ak.array_equal(
        ak.from_numpy(res_all_none), all_none, same_content_types=False
    )

    unitless = ak.Array(
        ak.contents.IndexedOptionArray(
            ak.index.Index64(np.array([0, -1], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([5], dtype=np.int64).view("timedelta64")),
        )
    )
    res_unitless = ak.to_numpy(unitless, allow_missing=True)
    assert isinstance(res_unitless, np.ma.MaskedArray)
    assert res_unitless.dtype == np.dtype("timedelta64")
    np.testing.assert_array_equal(res_unitless.mask, [False, True])
    assert ak.array_equal(
        ak.from_numpy(res_unitless), unitless, same_content_types=False
    )


@pytest.mark.parametrize("valid_when", [True, False])
def test_bytemasked_timedelta64_to_numpy(valid_when):
    mask = np.array([True, False, True] if valid_when else [False, True, False])
    layout = ak.contents.ByteMaskedArray(
        ak.index.Index8(mask),
        ak.contents.NumpyArray(np.array([10, 20, 30], dtype="timedelta64[s]")),
        valid_when=valid_when,
    )
    arr = ak.Array(layout)
    result = ak.to_numpy(arr, allow_missing=True)

    assert isinstance(result, np.ma.MaskedArray)
    assert result.dtype == np.dtype("timedelta64[s]")
    np.testing.assert_array_equal(result.mask, [False, True, False])
    assert result[0] == np.timedelta64(10, "s")
    assert result[2] == np.timedelta64(30, "s")
    assert ak.array_equal(ak.from_numpy(result), arr, same_content_types=False)


def test_bitmasked_and_unmasked_timedelta64_to_numpy():
    content = ak.contents.NumpyArray(np.array([4, 5, 6], dtype="timedelta64[ms]"))
    bitmasked = ak.Array(
        ak.contents.BitMaskedArray(
            ak.index.IndexU8(np.array([0b101], dtype=np.uint8)),
            content,
            valid_when=True,
            length=3,
            lsb_order=True,
        )
    )
    res_bit = ak.to_numpy(bitmasked, allow_missing=True)
    assert isinstance(res_bit, np.ma.MaskedArray)
    np.testing.assert_array_equal(res_bit.mask, [False, True, False])
    assert ak.array_equal(ak.from_numpy(res_bit), bitmasked, same_content_types=False)

    unmasked = ak.Array(ak.contents.UnmaskedArray(content))
    res_unmasked = ak.to_numpy(unmasked, allow_missing=True)
    assert isinstance(res_unmasked, np.ma.MaskedArray)
    np.testing.assert_array_equal(res_unmasked.mask, [False, False, False])
    assert ak.array_equal(
        ak.from_numpy(res_unmasked), unmasked, same_content_types=False
    )


def test_option_record_with_timedelta64_to_numpy():
    arr = ak.Array(
        [
            {"dt": np.timedelta64(2, "s"), "x": 1},
            None,
        ]
    )
    result = ak.to_numpy(arr, allow_missing=True)
    assert isinstance(result, np.ma.MaskedArray)
    assert result.dtype == np.dtype([("dt", "m8[s]"), ("x", "i8")])
    np.testing.assert_array_equal(result["dt"].mask, [False, True])
    np.testing.assert_array_equal(result["x"].mask, [False, True])
    assert result["dt"][0] == np.timedelta64(2, "s")
    assert result["x"][0] == 1

    field_opt = ak.Array([{"dt": np.timedelta64(1, "s")}, {"dt": None}])
    field_res = ak.to_numpy(field_opt, allow_missing=True)
    assert isinstance(field_res, np.ma.MaskedArray)
    assert field_res.dtype == np.dtype([("dt", "m8[s]")])
    np.testing.assert_array_equal(field_res["dt"].mask, [False, True])
    assert field_res["dt"][0] == np.timedelta64(1, "s")


def test_option_regular_timedelta64_to_numpy():
    arr = ak.Array(
        ak.contents.IndexedOptionArray(
            ak.index.Index64(np.array([0, -1], dtype=np.int64)),
            ak.contents.RegularArray(
                ak.contents.NumpyArray(np.array([1, 2, 3, 4], dtype="timedelta64[s]")),
                2,
            ),
        )
    )
    result = ak.to_numpy(arr, allow_missing=True)
    assert isinstance(result, np.ma.MaskedArray)
    assert result.shape == (2, 2)
    assert result.dtype == np.dtype("timedelta64[s]")
    np.testing.assert_array_equal(result.mask, [[False, False], [True, True]])
    assert result[0, 0] == np.timedelta64(1, "s")
    assert result[0, 1] == np.timedelta64(2, "s")
    roundtrip = ak.from_numpy(result)
    assert roundtrip.to_list() == [
        [np.timedelta64(1, "s"), np.timedelta64(2, "s")],
        [None, None],
    ]
    reconverted = ak.to_numpy(roundtrip, allow_missing=True)
    assert reconverted.dtype == result.dtype
    np.testing.assert_array_equal(reconverted, result)
    np.testing.assert_array_equal(
        np.ma.getmaskarray(reconverted), np.ma.getmaskarray(result)
    )


def test_option_timedelta64_disallow_missing():
    arr = ak.Array([np.timedelta64(1, "s"), None])
    with pytest.raises(ValueError, match="allow_missing"):
        ak.to_numpy(arr, allow_missing=False)
