# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

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
    assert ak.all(named_array < 4, axis=0).named_axis == ak.all(named_array < 4, axis="events").named_axis == ("jets",)
    assert ak.all(named_array < 4, axis=1).named_axis == ak.all(named_array < 4, axis="jets").named_axis == ("events",)
    assert ak.all(named_array < 4, axis=None).named_axis == (None,)


def test_named_axis_ak_almost_equal():
    array1 = array2 = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array1 = named_array2 = ak.with_named_axis(array1, named_axis=("events", "jets"))

    assert ak.almost_equal(array1, array2, check_named_axis=False) == ak.almost_equal(named_array1, named_array2, check_named_axis=False) == True
    assert ak.almost_equal(array1, array2, check_named_axis=True) == ak.almost_equal(named_array1, named_array2, check_named_axis=True) == True

    assert ak.almost_equal(named_array1, array1, check_named_axis=False) == True
    assert ak.almost_equal(named_array1, array1, check_named_axis=True) == True

    named_array3 = ak.with_named_axis(array1, named_axis=("events", "muons"))
    assert ak.almost_equal(named_array1, named_array3, check_named_axis=False) == True
    assert ak.almost_equal(named_array1, named_array3, check_named_axis=True) == False


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
    assert ak.any(named_array < 4, axis=0).named_axis == ak.any(named_array < 4, axis="events").named_axis == ("jets",)
    assert ak.any(named_array < 4, axis=1).named_axis == ak.any(named_array < 4, axis="jets").named_axis == ("events",)
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
    assert ak.all(ak.argmax(array, axis=0, keepdims=True) == ak.argmax(named_array, axis="events", keepdims=True))
    assert ak.all(ak.argmax(array, axis=1, keepdims=True) == ak.argmax(named_array, axis="jets", keepdims=True))
    assert ak.all(ak.argmax(array, axis=None) == ak.argmax(named_array, axis=None))

    # check that result axis names are correctly propagated
    assert ak.argmax(named_array, axis=0).named_axis == ak.argmax(named_array, axis="events").named_axis == ("jets",)
    assert ak.argmax(named_array, axis=1).named_axis == ak.argmax(named_array, axis="jets").named_axis == ("events",)
    assert ak.argmax(named_array, axis=0, keepdims=True).named_axis == ak.argmax(named_array, axis="events", keepdims=True).named_axis == ("events", "jets",)
    assert ak.argmax(named_array, axis=1, keepdims=True).named_axis == ak.argmax(named_array, axis="jets", keepdims=True).named_axis == ("events", "jets")
    assert ak.argmax(named_array, axis=None).named_axis == (None,)


def test_named_axis_ak_argmin():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("events", "jets"))

    # first check that they work the same
    assert ak.all(ak.argmin(array, axis=0) == ak.argmin(named_array, axis="events"))
    assert ak.all(ak.argmin(array, axis=1) == ak.argmin(named_array, axis="jets"))
    assert ak.all(ak.argmin(array, axis=0, keepdims=True) == ak.argmin(named_array, axis="events", keepdims=True))
    assert ak.all(ak.argmin(array, axis=1, keepdims=True) == ak.argmin(named_array, axis="jets", keepdims=True))
    assert ak.all(ak.argmin(array, axis=None) == ak.argmin(named_array, axis=None))

    # check that result axis names are correctly propagated
    assert ak.argmin(named_array, axis=0).named_axis == ak.argmin(named_array, axis="events").named_axis == ("jets",)
    assert ak.argmin(named_array, axis=1).named_axis == ak.argmin(named_array, axis="jets").named_axis == ("events",)
    assert ak.argmin(named_array, axis=0, keepdims=True).named_axis == ak.argmin(named_array, axis="events", keepdims=True).named_axis == ("events", "jets",)
    assert ak.argmin(named_array, axis=1, keepdims=True).named_axis == ak.argmin(named_array, axis="jets", keepdims=True).named_axis == ("events", "jets")
    assert ak.argmin(named_array, axis=None).named_axis == (None,)


def test_named_axis_ak_argsort():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("events", "jets"))

    # first check that they work the same
    assert ak.all(ak.argsort(array, axis=0) == ak.argsort(named_array, axis="events"))
    assert ak.all(ak.argsort(array, axis=1) == ak.argsort(named_array, axis="jets"))

    # check that result axis names are correctly propagated
    assert ak.argsort(named_array, axis=0).named_axis == ak.argsort(named_array, axis="events").named_axis == ("events", "jets")
    assert ak.argsort(named_array, axis=1).named_axis == ak.argsort(named_array, axis="jets").named_axis == ("events", "jets")


def test_named_axis_ak_array_equal():
    array1 = array2 = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array1 = named_array2 = ak.with_named_axis(array1, named_axis=("events", "jets"))

    assert ak.array_equal(array1, array2, check_named_axis=False) == ak.array_equal(named_array1, named_array2, check_named_axis=False) == True
    assert ak.array_equal(array1, array2, check_named_axis=True) == ak.array_equal(named_array1, named_array2, check_named_axis=True) == True

    assert ak.array_equal(named_array1, array1, check_named_axis=False) == True
    assert ak.array_equal(named_array1, array1, check_named_axis=True) == True

    named_array3 = ak.with_named_axis(array1, named_axis=("events", "muons"))
    assert ak.array_equal(named_array1, named_array3, check_named_axis=False) == True
    assert ak.array_equal(named_array1, named_array3, check_named_axis=True) == False


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
    assert True


def test_named_axis_ak_count():
    assert True


def test_named_axis_ak_count_nonzero():
    assert True


def test_named_axis_ak_covar():
    assert True


def test_named_axis_ak_drop_none():
    assert True


def test_named_axis_ak_enforce_type():
    assert True


def test_named_axis_ak_fields():
    assert True


def test_named_axis_ak_fill_none():
    assert True


def test_named_axis_ak_firsts():
    assert True


def test_named_axis_ak_flatten():
    assert True


def test_named_axis_ak_from_arrow():
    assert True


def test_named_axis_ak_from_arrow_schema():
    assert True


def test_named_axis_ak_from_avro_file():
    assert True


def test_named_axis_ak_from_buffers():
    assert True


def test_named_axis_ak_from_categorical():
    assert True


def test_named_axis_ak_from_cupy():
    assert True


def test_named_axis_ak_from_dlpack():
    assert True


def test_named_axis_ak_from_feather():
    assert True


def test_named_axis_ak_from_iter():
    assert True


def test_named_axis_ak_from_jax():
    assert True


def test_named_axis_ak_from_json():
    assert True


def test_named_axis_ak_from_numpy():
    assert True


def test_named_axis_ak_from_parquet():
    assert True


def test_named_axis_ak_from_raggedtensor():
    assert True


def test_named_axis_ak_from_rdataframe():
    assert True


def test_named_axis_ak_from_regular():
    assert True


def test_named_axis_ak_full_like():
    assert True


def test_named_axis_ak_imag():
    assert True


def test_named_axis_ak_is_categorical():
    assert True


def test_named_axis_ak_is_none():
    assert True


def test_named_axis_ak_is_tuple():
    assert True


def test_named_axis_ak_is_valid():
    assert True


def test_named_axis_ak_isclose():
    assert True


def test_named_axis_ak_linear_fit():
    assert True


def test_named_axis_ak_local_index():
    assert True


def test_named_axis_ak_mask():
    assert True


def test_named_axis_ak_max():
    array = ak.Array([[1, 2], [3], [], [4, 5, 6]])

    named_array = ak.with_named_axis(array, named_axis=("events", "jets"))

    # first check that they work the same
    assert ak.all(ak.max(array, axis=0) == ak.max(named_array, axis="events"))
    assert ak.all(ak.max(array, axis=1) == ak.max(named_array, axis="jets"))

    # check that result axis names are correctly propagated
    assert ak.max(named_array, axis=0).named_axis == ak.max(named_array, axis="events").named_axis == ("jets",)
    assert ak.max(named_array, axis=1).named_axis == ak.max(named_array, axis="jets").named_axis == ("events",)
    assert ak.max(named_array, axis=None).named_axis == (None,)


def test_named_axis_ak_mean():
    assert True


def test_named_axis_ak_merge_option_of_records():
    assert True


def test_named_axis_ak_merge_union_of_records():
    assert True


def test_named_axis_ak_metadata_from_parquet():
    assert True


def test_named_axis_ak_min():
    assert True


def test_named_axis_ak_moment():
    assert True


def test_named_axis_ak_nan_to_none():
    assert True


def test_named_axis_ak_nan_to_num():
    assert True


def test_named_axis_ak_num():
    assert True


def test_named_axis_ak_ones_like():
    assert True


def test_named_axis_ak_pad_none():
    assert True


def test_named_axis_ak_parameters():
    assert True


def test_named_axis_ak_prod():
    assert True


def test_named_axis_ak_ptp():
    assert True


def test_named_axis_ak_ravel():
    assert True


def test_named_axis_ak_real():
    assert True


def test_named_axis_ak_round():
    assert True


def test_named_axis_ak_run_lengths():
    assert True


def test_named_axis_ak_singletons():
    assert True


def test_named_axis_ak_softmax():
    assert True


def test_named_axis_ak_sort():
    assert True


def test_named_axis_ak_std():
    assert True


def test_named_axis_ak_strings_astype():
    assert True


def test_named_axis_ak_sum():
    assert True


def test_named_axis_ak_to_arrow():
    assert True


def test_named_axis_ak_to_arrow_table():
    assert True


def test_named_axis_ak_to_backend():
    assert True


def test_named_axis_ak_to_buffers():
    assert True


def test_named_axis_ak_to_cupy():
    assert True


def test_named_axis_ak_to_dataframe():
    assert True


def test_named_axis_ak_to_feather():
    assert True


def test_named_axis_ak_to_jax():
    assert True


def test_named_axis_ak_to_json():
    assert True


def test_named_axis_ak_to_layout():
    assert True


def test_named_axis_ak_to_list():
    assert True


def test_named_axis_ak_to_numpy():
    assert True


def test_named_axis_ak_to_packed():
    assert True


def test_named_axis_ak_to_parquet():
    assert True


def test_named_axis_ak_to_parquet_dataset():
    assert True


def test_named_axis_ak_to_parquet_row_groups():
    assert True


def test_named_axis_ak_to_raggedtensor():
    assert True


def test_named_axis_ak_to_rdataframe():
    assert True


def test_named_axis_ak_to_regular():
    assert True


def test_named_axis_ak_transform():
    assert True


def test_named_axis_ak_type():
    assert True


def test_named_axis_ak_unflatten():
    assert True


def test_named_axis_ak_unzip():
    assert True


def test_named_axis_ak_validity_error():
    assert True


def test_named_axis_ak_values_astype():
    assert True


def test_named_axis_ak_var():
    assert True


def test_named_axis_ak_where():
    assert True


def test_named_axis_ak_with_field():
    assert True


def test_named_axis_ak_with_name():
    assert True


def test_named_axis_ak_with_named_axis():
    assert True


def test_named_axis_ak_with_parameter():
    assert True


def test_named_axis_ak_without_field():
    assert True


def test_named_axis_ak_without_parameters():
    assert True


def test_named_axis_ak_zeros_like():
    assert True


def test_named_axis_ak_zip():
    assert True
