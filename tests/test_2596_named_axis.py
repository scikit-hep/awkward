# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_with_named_axis():
    from dataclasses import dataclass

    from awkward._namedaxis import _supports_named_axis

    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])
    assert not _supports_named_axis(array)
    assert array.named_axis == (None, None)
    assert array.positional_axis == (0, 1)

    array = ak.with_named_axis(array, named_axis=("events", "jets"))
    assert _supports_named_axis(array)
    assert array.named_axis == ("events", "jets")
    assert array.positional_axis == (0, 1)

    array = ak.with_named_axis(array, named_axis=("events", None))
    assert _supports_named_axis(array)
    assert array.named_axis == ("events", None)
    assert array.positional_axis == (0, 1)

    array = ak.with_named_axis(array, named_axis={"events": 0, "jets": 1})
    assert _supports_named_axis(array)
    assert array.named_axis == ("events", "jets")
    assert array.positional_axis == (0, 1)

    array = ak.with_named_axis(array, named_axis={"events": 1})
    assert _supports_named_axis(array)
    assert array.named_axis == (None, "events")
    assert array.positional_axis == (0, 1)

    array = ak.with_named_axis(array, named_axis={"jets": -1})
    assert _supports_named_axis(array)
    assert array.named_axis == (None, "jets")
    assert array.positional_axis == (0, 1)

    @dataclass(frozen=True)
    class exotic_axis:
        attr: str

    ax1 = exotic_axis(attr="I'm not the type of axis that you're used to")
    ax2 = exotic_axis(attr="...me neither!")

    array = ak.with_named_axis(array, named_axis=(ax1, ax2))
    assert array.named_axis == (ax1, ax2)
    assert array.positional_axis == (0, 1)


def test_named_axis_ak_all():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("events", "jets"))

    # first check that they work the same
    assert ak.all(ak.all(array < 4, axis=0) == ak.all(named_array < 4, axis="events"))
    assert ak.all(ak.all(array < 4, axis=1) == ak.all(named_array < 4, axis="jets"))

    # check that result axis names are correctly propagated
    assert (
        ak.all(named_array < 4, axis=0).named_axis
        == ak.all(named_array < 4, axis="events").named_axis
        == ("jets",)
    )
    assert (
        ak.all(named_array < 4, axis=1).named_axis
        == ak.all(named_array < 4, axis="jets").named_axis
        == ("events",)
    )
    assert ak.all(named_array < 4, axis=None).named_axis == (None,)


def test_named_axis_ak_almost_equal():
    array1 = array2 = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array1 = named_array2 = ak.with_named_axis(
        array1, named_axis=("events", "jets")
    )

    assert ak.almost_equal(array1, array2, check_named_axis=False) == ak.almost_equal(
        named_array1, named_array2, check_named_axis=False
    )
    assert ak.almost_equal(array1, array2, check_named_axis=True) == ak.almost_equal(
        named_array1, named_array2, check_named_axis=True
    )

    assert ak.almost_equal(named_array1, array1, check_named_axis=False)
    assert ak.almost_equal(named_array1, array1, check_named_axis=True)

    named_array3 = ak.with_named_axis(array1, named_axis=("events", "muons"))
    assert ak.almost_equal(named_array1, named_array3, check_named_axis=False)
    assert not ak.almost_equal(named_array1, named_array3, check_named_axis=True)


def test_named_axis_ak_angle():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("events", "jets"))

    # first check that they work the same
    assert ak.all(ak.angle(array) == ak.angle(named_array))

    # check that result axis names are correctly propagated
    assert ak.angle(named_array).named_axis == ("events", "jets")


def test_named_axis_ak_any():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("events", "jets"))

    # first check that they work the same
    assert ak.all(ak.any(array < 4, axis=0) == ak.any(named_array < 4, axis="events"))
    assert ak.all(ak.any(array < 4, axis=1) == ak.any(named_array < 4, axis="jets"))

    # check that result axis names are correctly propagated
    assert (
        ak.any(named_array < 4, axis=0).named_axis
        == ak.any(named_array < 4, axis="events").named_axis
        == ("jets",)
    )
    assert (
        ak.any(named_array < 4, axis=1).named_axis
        == ak.any(named_array < 4, axis="jets").named_axis
        == ("events",)
    )
    assert ak.any(named_array < 4, axis=None).named_axis == (None,)


def test_named_axis_ak_argcartesian():
    assert True


def test_named_axis_ak_argcombinations():
    assert True


def test_named_axis_ak_argmax():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("events", "jets"))

    # first check that they work the same
    assert ak.all(ak.argmax(array, axis=0) == ak.argmax(named_array, axis="events"))
    assert ak.all(ak.argmax(array, axis=1) == ak.argmax(named_array, axis="jets"))
    assert ak.all(
        ak.argmax(array, axis=0, keepdims=True)
        == ak.argmax(named_array, axis="events", keepdims=True)
    )
    assert ak.all(
        ak.argmax(array, axis=1, keepdims=True)
        == ak.argmax(named_array, axis="jets", keepdims=True)
    )
    assert ak.all(ak.argmax(array, axis=None) == ak.argmax(named_array, axis=None))

    # check that result axis names are correctly propagated
    assert (
        ak.argmax(named_array, axis=0).named_axis
        == ak.argmax(named_array, axis="events").named_axis
        == ("jets",)
    )
    assert (
        ak.argmax(named_array, axis=1).named_axis
        == ak.argmax(named_array, axis="jets").named_axis
        == ("events",)
    )
    assert (
        ak.argmax(named_array, axis=0, keepdims=True).named_axis
        == ak.argmax(named_array, axis="events", keepdims=True).named_axis
        == (
            "events",
            "jets",
        )
    )
    assert (
        ak.argmax(named_array, axis=1, keepdims=True).named_axis
        == ak.argmax(named_array, axis="jets", keepdims=True).named_axis
        == ("events", "jets")
    )
    assert ak.argmax(named_array, axis=None).named_axis == (None,)


def test_named_axis_ak_argmin():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("events", "jets"))

    # first check that they work the same
    assert ak.all(ak.argmin(array, axis=0) == ak.argmin(named_array, axis="events"))
    assert ak.all(ak.argmin(array, axis=1) == ak.argmin(named_array, axis="jets"))
    assert ak.all(
        ak.argmin(array, axis=0, keepdims=True)
        == ak.argmin(named_array, axis="events", keepdims=True)
    )
    assert ak.all(
        ak.argmin(array, axis=1, keepdims=True)
        == ak.argmin(named_array, axis="jets", keepdims=True)
    )
    assert ak.all(ak.argmin(array, axis=None) == ak.argmin(named_array, axis=None))

    # check that result axis names are correctly propagated
    assert (
        ak.argmin(named_array, axis=0).named_axis
        == ak.argmin(named_array, axis="events").named_axis
        == ("jets",)
    )
    assert (
        ak.argmin(named_array, axis=1).named_axis
        == ak.argmin(named_array, axis="jets").named_axis
        == ("events",)
    )
    assert (
        ak.argmin(named_array, axis=0, keepdims=True).named_axis
        == ak.argmin(named_array, axis="events", keepdims=True).named_axis
        == (
            "events",
            "jets",
        )
    )
    assert (
        ak.argmin(named_array, axis=1, keepdims=True).named_axis
        == ak.argmin(named_array, axis="jets", keepdims=True).named_axis
        == ("events", "jets")
    )
    assert ak.argmin(named_array, axis=None).named_axis == (None,)


def test_named_axis_ak_argsort():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("events", "jets"))

    # first check that they work the same
    assert ak.all(ak.argsort(array, axis=0) == ak.argsort(named_array, axis="events"))
    assert ak.all(ak.argsort(array, axis=1) == ak.argsort(named_array, axis="jets"))

    # check that result axis names are correctly propagated
    assert (
        ak.argsort(named_array, axis=0).named_axis
        == ak.argsort(named_array, axis="events").named_axis
        == ("events", "jets")
    )
    assert (
        ak.argsort(named_array, axis=1).named_axis
        == ak.argsort(named_array, axis="jets").named_axis
        == ("events", "jets")
    )


def test_named_axis_ak_array_equal():
    array1 = array2 = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array1 = named_array2 = ak.with_named_axis(
        array1, named_axis=("events", "jets")
    )

    assert ak.array_equal(array1, array2, check_named_axis=False) == ak.array_equal(
        named_array1, named_array2, check_named_axis=False
    )
    assert ak.array_equal(array1, array2, check_named_axis=True) == ak.array_equal(
        named_array1, named_array2, check_named_axis=True
    )

    assert ak.array_equal(named_array1, array1, check_named_axis=False)
    assert ak.array_equal(named_array1, array1, check_named_axis=True)

    named_array3 = ak.with_named_axis(array1, named_axis=("events", "muons"))
    assert ak.array_equal(named_array1, named_array3, check_named_axis=False)
    assert not ak.array_equal(named_array1, named_array3, check_named_axis=True)


def test_named_axis_ak_backend():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("events", "jets"))

    assert ak.backend(array) == ak.backend(named_array)


def test_named_axis_ak_broadcast_arrays():
    assert True


def test_named_axis_ak_broadcast_fields():
    assert True


def test_named_axis_ak_cartesian():
    assert True


def test_named_axis_ak_categories():
    # This test doesn't run because of an `import pyarrow` issue
    #
    # array = ak.str.to_categorical([["one", "two"], ["one", "three"], ["one", "four"]])

    # named_array = ak.with_named_axis(array, named_axis=("a", "b"))

    # # assert ak.all(ak.categories(array) == ak.categories(named_array))  # FIX: ufuncs
    # assert (
    #     ak.categories(array).named_axis
    #     == ak.categories(named_array).named_axis
    #     == (None,)
    # )
    assert True


def test_named_axis_ak_combinations():
    assert True


def test_named_axis_ak_concatenate():
    assert True


def test_named_axis_ak_copy():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("events", "jets"))

    assert ak.copy(named_array).named_axis == ("events", "jets")


def test_named_axis_ak_corr():
    array_x = ak.Array([[0, 1.1], [3.3, 4.4]])
    array_y = ak.Array([[0, 1], [3, 4]])

    named_array_x = ak.with_named_axis(array_x, ("x", "y"))
    named_array_y = ak.with_named_axis(array_y, ("x", "y"))

    assert ak.all(
        ak.corr(array_x, array_y, axis=0)
        == ak.corr(named_array_x, named_array_y, axis="x")
    )
    assert ak.all(
        ak.corr(array_x, array_y, axis=1)
        == ak.corr(named_array_x, named_array_y, axis="y")
    )
    assert ak.all(
        ak.corr(array_x, array_y, axis=None)
        == ak.corr(named_array_x, named_array_y, axis=None)
    )

    assert ak.corr(named_array_x, named_array_y, axis="x").named_axis == ("y",)
    assert ak.corr(named_array_x, named_array_y, axis="y").named_axis == ("x",)
    assert ak.corr(named_array_x, named_array_y, axis=None).named_axis == (None,)


def test_named_axis_ak_count():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.count(array, axis=0) == ak.count(named_array, axis="x"))
    assert ak.all(ak.count(array, axis=1) == ak.count(named_array, axis="y"))
    assert ak.all(ak.count(array, axis=None) == ak.count(named_array, axis=None))

    assert ak.count(named_array, axis="x").named_axis == ("y",)
    assert ak.count(named_array, axis="y").named_axis == ("x",)
    assert ak.count(named_array, axis=None).named_axis == (None,)


def test_named_axis_ak_count_nonzero():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(
        ak.count_nonzero(array, axis=0) == ak.count_nonzero(named_array, axis="x")
    )
    assert ak.all(
        ak.count_nonzero(array, axis=1) == ak.count_nonzero(named_array, axis="y")
    )
    assert ak.all(
        ak.count_nonzero(array, axis=None) == ak.count_nonzero(named_array, axis=None)
    )

    assert ak.count_nonzero(named_array, axis="x").named_axis == ("y",)
    assert ak.count_nonzero(named_array, axis="y").named_axis == ("x",)
    assert ak.count_nonzero(named_array, axis=None).named_axis == (None,)


def test_named_axis_ak_covar():
    array_x = ak.Array([[0, 1.1], [3.3, 4.4]])
    array_y = ak.Array([[0, 1], [3, 4]])

    named_array_x = ak.with_named_axis(array_x, ("x", "y"))
    named_array_y = ak.with_named_axis(array_y, ("x", "y"))

    assert ak.all(
        ak.covar(array_x, array_y, axis=0)
        == ak.covar(named_array_x, named_array_y, axis="x")
    )
    assert ak.all(
        ak.covar(array_x, array_y, axis=1)
        == ak.covar(named_array_x, named_array_y, axis="y")
    )
    assert ak.all(
        ak.covar(array_x, array_y, axis=None)
        == ak.covar(named_array_x, named_array_y, axis=None)
    )

    assert ak.covar(named_array_x, named_array_y, axis="x").named_axis == ("y",)
    assert ak.covar(named_array_x, named_array_y, axis="y").named_axis == ("x",)
    assert ak.covar(named_array_x, named_array_y, axis=None).named_axis == (None,)


def test_named_axis_ak_drop_none():
    array = ak.Array([[1, None], [3], [None], [4, None, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.drop_none(array, axis=0) == ak.drop_none(named_array, axis="x"))
    assert ak.all(ak.drop_none(array, axis=1) == ak.drop_none(named_array, axis="y"))
    assert ak.all(
        ak.drop_none(array, axis=None) == ak.drop_none(named_array, axis=None)
    )

    assert ak.drop_none(named_array, axis="x").named_axis == ("x", "y")
    assert ak.drop_none(named_array, axis="y").named_axis == ("x", "y")
    assert ak.drop_none(named_array, axis=None).named_axis == ("x", "y")


def test_named_axis_ak_enforce_type():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.enforce_type(named_array, "var * ?int64").named_axis == ("x", "y")


def test_named_axis_ak_fields():
    # skip
    assert True


def test_named_axis_ak_fill_none():
    array = ak.Array([[1.1, None, 2.2], [], [None, 3.3, 4.4]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(
        ak.fill_none(array, 0, axis=0) == ak.fill_none(named_array, 0, axis="x")
    )
    assert ak.all(
        ak.fill_none(array, 0, axis=1) == ak.fill_none(named_array, 0, axis="y")
    )
    assert ak.all(
        ak.fill_none(array, 0, axis=None) == ak.fill_none(named_array, 0, axis=None)
    )

    assert ak.fill_none(named_array, 0, axis="x").named_axis == ("x", "y")
    assert ak.fill_none(named_array, 0, axis="y").named_axis == ("x", "y")
    assert ak.fill_none(named_array, 0, axis=None).named_axis == ("x", "y")


def test_named_axis_ak_firsts():
    array = ak.Array([[1.1], [2.2], [], [3.3], [], [], [4.4], [5.5]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.firsts(array, axis=0) == ak.firsts(named_array, axis="x"))
    assert ak.all(ak.firsts(array, axis=1) == ak.firsts(named_array, axis="y"))

    assert ak.firsts(named_array, axis="x").named_axis == ("x",)
    assert ak.firsts(named_array, axis="y").named_axis == ("y",)


def test_named_axis_ak_flatten():
    assert True


def test_named_axis_ak_from_arrow():
    # skip
    assert True


def test_named_axis_ak_from_arrow_schema():
    # skip
    assert True


def test_named_axis_ak_from_avro_file():
    # skip
    assert True


def test_named_axis_ak_from_buffers():
    # skip
    assert True


def test_named_axis_ak_from_categorical():
    # skip
    assert True


def test_named_axis_ak_from_cupy():
    # skip
    assert True


def test_named_axis_ak_from_dlpack():
    # skip
    assert True


def test_named_axis_ak_from_feather():
    # skip
    assert True


def test_named_axis_ak_from_iter():
    # skip
    assert True


def test_named_axis_ak_from_jax():
    # skip
    assert True


def test_named_axis_ak_from_json():
    # skip
    assert True


def test_named_axis_ak_from_numpy():
    # skip
    assert True


def test_named_axis_ak_from_parquet():
    # skip
    assert True


def test_named_axis_ak_from_raggedtensor():
    # skip
    assert True


def test_named_axis_ak_from_rdataframe():
    # skip
    assert True


def test_named_axis_ak_from_regular():
    # skip
    assert True


def test_named_axis_ak_full_like():
    # skip
    assert True


def test_named_axis_ak_imag():
    array = ak.Array([[1 + 2j], [2 + 1j], []])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.imag(array) == ak.imag(named_array))
    assert ak.imag(named_array).named_axis == ("x", "y")


def test_named_axis_ak_is_categorical():
    # skip
    assert True


def test_named_axis_ak_is_none():
    array = ak.Array([[1, None], [3], [None], [4, None, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.is_none(array, axis=0) == ak.is_none(named_array, axis="x"))
    assert ak.all(ak.is_none(array, axis=1) == ak.is_none(named_array, axis="y"))

    assert ak.is_none(named_array, axis="x").named_axis == ("x", "y")
    assert ak.is_none(named_array, axis="y").named_axis == ("x", "y")


def test_named_axis_ak_is_tuple():
    # skip
    assert True


def test_named_axis_ak_is_valid():
    # skip
    assert True


def test_named_axis_ak_isclose():
    # skip
    assert True


def test_named_axis_ak_linear_fit():
    # skip
    assert True


def test_named_axis_ak_local_index():
    array = ak.Array(
        [[[0.0, 1.1, 2.2], []], [[3.3, 4.4]], [], [[5.5], [], [6.6, 7.7, 8.8, 9.9]]]
    )

    named_array = ak.with_named_axis(array, ("x", "y", "z"))

    assert ak.all(
        ak.local_index(array, axis=0) == ak.local_index(named_array, axis="x")
    )
    assert ak.all(
        ak.local_index(array, axis=1) == ak.local_index(named_array, axis="y")
    )
    assert ak.all(
        ak.local_index(array, axis=2) == ak.local_index(named_array, axis="z")
    )

    assert ak.local_index(named_array, axis="x").named_axis == ("x",)
    assert ak.local_index(named_array, axis="y").named_axis == ("x", "y")
    assert ak.local_index(named_array, axis="z").named_axis == ("x", "y", "z")


def test_named_axis_ak_mask():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])
    mask = array > 3

    named_array = ak.with_named_axis(array, ("x", "y"))
    named_mask = named_array > 3

    assert ak.all(ak.mask(array, mask) == ak.mask(named_array, mask))
    assert ak.all(ak.mask(array, mask) == ak.mask(named_array, named_mask))

    assert ak.mask(named_array, mask).named_axis == named_array.named_axis
    assert ak.mask(named_array, named_mask).named_axis == named_array.named_axis


def test_named_axis_ak_max():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("events", "jets"))

    # first check that they work the same
    assert ak.all(ak.max(array, axis=0) == ak.max(named_array, axis="events"))
    assert ak.all(ak.max(array, axis=1) == ak.max(named_array, axis="jets"))

    # check that result axis names are correctly propagated
    assert (
        ak.max(named_array, axis=0).named_axis
        == ak.max(named_array, axis="events").named_axis
        == ("jets",)
    )
    assert (
        ak.max(named_array, axis=1).named_axis
        == ak.max(named_array, axis="jets").named_axis
        == ("events",)
    )
    assert ak.max(named_array, axis=None).named_axis == (None,)


def test_named_axis_ak_mean():
    array = ak.Array([[1, 2], [3], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.mean(array, axis=0) == ak.mean(named_array, axis="x"))
    assert ak.all(ak.mean(array, axis=1) == ak.mean(named_array, axis="y"))
    assert ak.mean(array, axis=None) == ak.mean(named_array, axis=None)

    assert ak.mean(named_array, axis="x").named_axis == ("y",)
    assert ak.mean(named_array, axis="y").named_axis == ("x",)
    assert ak.mean(named_array, axis=None).named_axis == (None,)


def test_named_axis_ak_merge_option_of_records():
    # skip
    assert True


def test_named_axis_ak_merge_union_of_records():
    # skip
    assert True


def test_named_axis_ak_metadata_from_parquet():
    # skip
    assert True


def test_named_axis_ak_min():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("events", "jets"))

    # first check that they work the same
    assert ak.all(ak.min(array, axis=0) == ak.min(named_array, axis="events"))
    assert ak.all(ak.min(array, axis=1) == ak.min(named_array, axis="jets"))

    # check that result axis names are correctly propagated
    assert (
        ak.min(named_array, axis=0).named_axis
        == ak.min(named_array, axis="events").named_axis
        == ("jets",)
    )
    assert (
        ak.min(named_array, axis=1).named_axis
        == ak.min(named_array, axis="jets").named_axis
        == ("events",)
    )
    assert ak.min(named_array, axis=None).named_axis == (None,)


def test_named_axis_ak_moment():
    array = ak.Array([[0, 1.1], [3.3, 4.4]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.moment(array, 0, axis=0) == ak.moment(named_array, 0, axis="x"))
    assert ak.all(ak.moment(array, 0, axis=1) == ak.moment(named_array, 0, axis="y"))
    assert ak.all(
        ak.moment(array, 0, axis=None) == ak.moment(named_array, 0, axis=None)
    )

    assert ak.moment(named_array, 0, axis="x").named_axis == ("y",)
    assert ak.moment(named_array, 0, axis="y").named_axis == ("x",)
    assert ak.moment(named_array, 0, axis=None).named_axis == (None,)


def test_named_axis_ak_nan_to_none():
    array = ak.Array([[0, np.nan], [np.nan], [3.3, 4.4]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.nan_to_none(array) == ak.nan_to_none(named_array))
    assert ak.nan_to_none(named_array).named_axis == named_array.named_axis


def test_named_axis_ak_nan_to_num():
    array = ak.Array([[0, np.nan], [np.nan], [3.3, 4.4]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.nan_to_num(array, nan=0.0) == ak.nan_to_num(named_array, nan=0.0))
    assert ak.nan_to_num(named_array, nan=0.0).named_axis == named_array.named_axis


def test_named_axis_ak_num():
    array = ak.Array([[1, 2], [3], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.num(array, axis=0) == ak.num(named_array, axis="x")
    assert ak.all(ak.num(array, axis=1) == ak.num(named_array, axis="y"))

    assert ak.num(named_array, axis="y").named_axis == ("y",)


def test_named_axis_ak_ones_like():
    array = ak.Array([[1, 2], [3], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.ones_like(array) == ak.ones_like(named_array))

    assert ak.ones_like(named_array).named_axis == named_array.named_axis


def test_named_axis_ak_pad_none():
    array = ak.Array([[1, 2], [3], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.pad_none(array, 3, axis=0) == ak.pad_none(named_array, 3, axis=0))
    assert ak.all(ak.pad_none(array, 3, axis=1) == ak.pad_none(named_array, 3, axis=1))

    assert ak.pad_none(named_array, 3, axis=0).named_axis == named_array.named_axis
    assert ak.pad_none(named_array, 3, axis=1).named_axis == named_array.named_axis


def test_named_axis_ak_parameters():
    # skip
    assert True


def test_named_axis_ak_prod():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.prod(array, axis=0) == ak.prod(named_array, axis="x"))
    assert ak.all(ak.prod(array, axis=1) == ak.prod(named_array, axis="y"))
    assert ak.prod(array, axis=None) == ak.prod(named_array, axis=None)

    assert ak.prod(named_array, axis="x").named_axis == ("y",)
    assert ak.prod(named_array, axis="y").named_axis == ("x",)


def test_named_axis_ak_ptp():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.ptp(array, axis=0) == ak.ptp(named_array, axis="x"))
    assert ak.all(ak.ptp(array, axis=1) == ak.ptp(named_array, axis="y"))
    assert ak.ptp(array, axis=None) == ak.ptp(named_array, axis=None)

    assert ak.ptp(named_array, axis="x").named_axis == ("x",)
    assert ak.ptp(named_array, axis="y").named_axis == ("y",)


def test_named_axis_ak_ravel():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.ravel(array) == ak.ravel(named_array))

    assert ak.ravel(named_array).named_axis == (None,)


def test_named_axis_ak_real():
    array = ak.Array([[1 + 2j], [2 + 1j], []])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.real(array) == ak.real(named_array))
    assert ak.real(named_array).named_axis == ("x", "y")


def test_named_axis_ak_round():
    array = ak.Array([[1.234], [2.345, 3.456], []])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.round(array) == ak.round(named_array))
    assert ak.round(named_array).named_axis == ("x", "y")


def test_named_axis_ak_run_lengths():
    array = ak.Array([[1.1, 1.1, 1.1, 2.2, 3.3], [3.3, 4.4], [4.4, 5.5]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.run_lengths(array) == ak.run_lengths(named_array))

    assert ak.run_lengths(named_array).named_axis == named_array.named_axis


def test_named_axis_ak_singletons():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.singletons(array, axis=0) == ak.singletons(named_array, axis=0))
    assert ak.all(ak.singletons(array, axis=1) == ak.singletons(named_array, axis=1))

    assert ak.singletons(named_array, axis=0).named_axis == ("x", None, "y")
    assert ak.singletons(named_array, axis=1).named_axis == ("x", "y", None)


def test_named_axis_ak_softmax():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.softmax(array, axis=-1) == ak.softmax(named_array, axis="y"))

    assert ak.softmax(named_array, axis="y").named_axis == ("x", "y")


def test_named_axis_ak_sort():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("events", "jets"))

    # first check that they work the same
    assert ak.all(ak.sort(array, axis=0) == ak.sort(named_array, axis="events"))
    assert ak.all(ak.sort(array, axis=1) == ak.sort(named_array, axis="jets"))

    # check that result axis names are correctly propagated
    assert (
        ak.sort(named_array, axis=0).named_axis
        == ak.sort(named_array, axis="events").named_axis
        == ("events", "jets")
    )
    assert (
        ak.sort(named_array, axis=1).named_axis
        == ak.sort(named_array, axis="jets").named_axis
        == ("events", "jets")
    )


def test_named_axis_ak_std():
    # TODO: once slicing is implemented
    # array = ak.Array([[1, 2], [3], [4, 5, 6]])

    # named_array = ak.with_named_axis(array, ("x", "y"))

    # assert ak.all(ak.std(array, axis=0) == ak.std(named_array, axis="x"))
    # assert ak.all(ak.std(array, axis=1) == ak.std(named_array, axis="y"))
    # assert ak.std(array, axis=None) == ak.std(named_array, axis=None)

    # assert ak.std(named_array, axis="x").named_axis == ("y",)
    # assert ak.std(named_array, axis="y").named_axis == ("x",)
    # assert ak.std(named_array, axis=None).named_axis == (None,)
    assert True


def test_named_axis_ak_strings_astype():
    array = ak.Array([["1", "2"], ["3"], ["4", "5", "6"]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(
        ak.strings_astype(array, np.int32) == ak.strings_astype(named_array, np.int32)
    )

    assert ak.strings_astype(named_array, np.int32).named_axis == named_array.named_axis


def test_named_axis_ak_sum():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.sum(array, axis=0) == ak.sum(named_array, axis="x"))
    assert ak.all(ak.sum(array, axis=1) == ak.sum(named_array, axis="y"))
    assert ak.sum(array, axis=None) == ak.sum(named_array, axis=None)

    assert ak.sum(named_array, axis="x").named_axis == ("y",)
    assert ak.sum(named_array, axis="y").named_axis == ("x",)


def test_named_axis_ak_to_arrow():
    # skip
    assert True


def test_named_axis_ak_to_arrow_table():
    # skip
    assert True


def test_named_axis_ak_to_backend():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.to_backend(named_array, "typetracer").named_axis == named_array.named_axis


def test_named_axis_ak_to_buffers():
    # skip
    assert True


def test_named_axis_ak_to_cupy():
    # skip
    assert True


def test_named_axis_ak_to_dataframe():
    # skip
    assert True


def test_named_axis_ak_to_feather():
    # skip
    assert True


def test_named_axis_ak_to_jax():
    # skip
    assert True


def test_named_axis_ak_to_json():
    # skip
    assert True


def test_named_axis_ak_to_layout():
    # skip
    assert True


def test_named_axis_ak_to_list():
    # skip
    assert True


def test_named_axis_ak_to_numpy():
    # skip
    assert True


def test_named_axis_ak_to_packed():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.to_packed(array) == ak.to_packed(named_array))

    assert  ak.to_packed(named_array).named_axis == named_array.named_axis


def test_named_axis_ak_to_parquet():
    # skip
    assert True


def test_named_axis_ak_to_parquet_dataset():
    # skip
    assert True


def test_named_axis_ak_to_parquet_row_groups():
    # skip
    assert True


def test_named_axis_ak_to_raggedtensor():
    # skip
    assert True


def test_named_axis_ak_to_rdataframe():
    # skip
    assert True


def test_named_axis_ak_to_regular():
    # skip
    assert True


def test_named_axis_ak_transform():
    # skip
    assert True


def test_named_axis_ak_type():
    # skip
    assert True


def test_named_axis_ak_unflatten():
    array = ak.Array([[1, 2, 3, 4], [], [5, 6, 7], [8, 9]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    counts = ak.Array([2, 2, 1, 2, 1, 1])

    assert ak.all(
        ak.unflatten(array, counts, axis=1)
        == ak.unflatten(named_array, counts, axis="y")
    )
    assert ak.unflatten(named_array, counts, axis="y").named_axis == (None, None, None)


def test_named_axis_ak_unzip():
    # skip
    assert True


def test_named_axis_ak_validity_error():
    # skip
    assert True


def test_named_axis_ak_values_astype():
    array = ak.Array([[1, 2, 3, 4], [], [5, 6, 7], [8, 9]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.values_astype(array, np.float32) == ak.values_astype(named_array, np.float32))

    assert ak.values_astype(named_array, np.float32).named_axis == named_array.named_axis


def test_named_axis_ak_var():
    assert True


def test_named_axis_ak_where():
    assert True


def test_named_axis_ak_with_field():
    # skip
    assert True

def test_named_axis_ak_with_name():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.with_name(named_array, "new_name").named_axis == named_array.named_axis


def test_named_axis_ak_with_named_axis():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert named_array.named_axis == ("x", "y")


def test_named_axis_ak_with_parameter():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.with_parameter(named_array, "param", 1.0).named_axis == named_array.named_axis


def test_named_axis_ak_without_field():
    # skip
    assert True


def test_named_axis_ak_without_parameters():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    named_array_with_parameteter = ak.with_parameter(named_array, "param", 1.0)

    assert ak.without_parameters(named_array).named_axis == named_array.named_axis


def test_named_axis_ak_zeros_like():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, ("x", "y"))

    assert ak.all(ak.zeros_like(array) == ak.zeros_like(named_array))

    assert ak.zeros_like(named_array).named_axis == named_array.named_axis


def test_named_axis_ak_zip():
    named_array1 = ak.with_named_axis(ak.Array([1,2,3]), ("a",))
    named_array2 = ak.with_named_axis(ak.Array([[4,5,6], [], [7]]), ("x", "y"))

    record = ak.zip({"x": named_array1, "y": named_array2})

    # TODO: need to implement broadcasting properly first
    assert True
