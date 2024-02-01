# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import packaging.version
import pytest

import awkward as ak
from awkward._pickle import use_builtin_reducer


@pytest.fixture
def array_pickler():
    import pickle

    with use_builtin_reducer():
        yield pickle


SOME_ATTRS = {"foo": "!SOME"}
OTHER_ATTRS = {"bar": "!OTHER", "foo": "!OTHER"}


def test_set_attrs():
    array = ak.Array([1, 2, 3])
    assert array.attrs == {}

    array.attrs = OTHER_ATTRS
    assert array.attrs is OTHER_ATTRS

    with pytest.raises(TypeError):
        array.attrs = "Hello world!"


def test_serialise_with_transient_attrs(array_pickler):
    attrs = {**SOME_ATTRS, "@transient_key": lambda: None}
    array = ak.Array([1, 2, 3], attrs=attrs)
    result = array_pickler.loads(array_pickler.dumps(array))
    assert result.attrs == SOME_ATTRS


def test_serialise_with_nonserialisable_attrs(array_pickler):
    attrs = {**SOME_ATTRS, "non_transient_key": lambda: None}
    array = ak.Array([1, 2, 3], attrs=attrs)
    with pytest.raises(
        (AttributeError, array_pickler.PicklingError), match=r"Can't pickle"
    ):
        array_pickler.loads(array_pickler.dumps(array))


def test_transient_metadata_persists():
    attrs = {**SOME_ATTRS, "@transient_key": lambda: None}
    array = ak.Array([[1, 2, 3]], attrs=attrs)
    num = ak.num(array)
    assert num.attrs is attrs


@pytest.mark.parametrize(
    "func",
    [
        ak.any,
        ak.min,
        ak.argmin,
        ak.sum,
        ak.ptp,
        ak.count_nonzero,
        lambda *args, **kwargs: ak.moment(*args, **kwargs, n=3),
        ak.argmax,
        ak.all,
        ak.mean,
        ak.max,
        ak.prod,
        ak.count,
    ],
)
def test_single_arg_ops(func):
    # Default no attrs
    assert func([[1, 2, 3, 4], [5], [10]], axis=-1, highlevel=True).attrs == {}
    # Carry from argument
    assert (
        func([[1, 2, 3, 4], [5], [10]], axis=-1, highlevel=True, attrs=SOME_ATTRS).attrs
        is SOME_ATTRS
    )
    # Carry from outer array
    array = ak.Array([[1, 2, 3, 4], [5], [10]], attrs=SOME_ATTRS)
    assert func(array, axis=-1, highlevel=True).attrs is SOME_ATTRS
    # Carry from argument exclusively
    assert func(array, axis=-1, highlevel=True, attrs=OTHER_ATTRS).attrs is OTHER_ATTRS


@pytest.mark.parametrize(
    "func",
    [
        # Predicates
        ak.str.is_alnum,
        ak.str.is_alpha,
        ak.str.is_ascii,
        ak.str.is_decimal,
        ak.str.is_digit,
        ak.str.is_lower,
        ak.str.is_numeric,
        ak.str.is_printable,
        ak.str.is_space,
        ak.str.is_title,
        ak.str.is_upper,
        # Transforms
        ak.str.capitalize,
        ak.str.lower,
        ak.str.upper,
        ak.str.reverse,
        ak.str.swapcase,
        ak.str.title,
        # Padding and trimming
        ak.str.ltrim_whitespace,
        ak.str.rtrim_whitespace,
        ak.str.trim_whitespace,
        ak.str.split_whitespace,
    ],
)
def test_string_operations_unary(func):
    pytest.importorskip("pyarrow")
    # Default no attrs
    assert (
        func(
            [["hello", "world!"], [], ["it's a beautiful day!"]],
            highlevel=True,
        ).attrs
        == {}
    )
    # Carry from argument
    assert (
        func(
            [["hello", "world!"], [], ["it's a beautiful day!"]],
            highlevel=True,
            attrs=SOME_ATTRS,
        ).attrs
        is SOME_ATTRS
    )
    # Carry from outer array
    array = ak.Array(
        [["hello", "world!"], [], ["it's a beautiful day!"]], attrs=SOME_ATTRS
    )
    assert func(array, highlevel=True).attrs is SOME_ATTRS
    # Carry from argument exclusively
    assert func(array, highlevel=True, attrs=OTHER_ATTRS).attrs is OTHER_ATTRS


@pytest.mark.parametrize(
    ("func", "arg"),
    [
        # Padding and trimming
        (ak.str.center, 10),
        (ak.str.lpad, 10),
        (ak.str.rpad, 10),
        (ak.str.ltrim, "hell"),
        (ak.str.rtrim, "hell"),
        (ak.str.trim, "hell"),
        # Containment
        (ak.str.count_substring, "hello"),
        (ak.str.count_substring_regex, "hello"),
        (ak.str.starts_with, "hello"),
        (ak.str.ends_with, "hello"),
        (ak.str.find_substring, "hello"),
        (ak.str.find_substring_regex, "hello"),
        (ak.str.match_like, "hello"),
        (ak.str.match_substring, "hello"),
        (ak.str.match_substring_regex, "hello"),
        # Slicing
        (ak.str.extract_regex, "hello"),
    ],
)
def test_string_operations_unary_with_arg(func, arg):
    pytest.importorskip("pyarrow")
    # Default no attrs
    assert (
        func(
            [["hello", "world!"], [], ["it's a beautiful day!"]],
            arg,
            highlevel=True,
        ).attrs
        == {}
    )
    # Carry from argument
    assert (
        func(
            [["hello", "world!"], [], ["it's a beautiful day!"]],
            arg,
            highlevel=True,
            attrs=SOME_ATTRS,
        ).attrs
        is SOME_ATTRS
    )
    # Carry from outer array
    array = ak.Array(
        [["hello", "world!"], [], ["it's a beautiful day!"]], attrs=SOME_ATTRS
    )
    assert func(array, arg, highlevel=True).attrs is SOME_ATTRS
    # Carry from argument exclusively
    assert func(array, arg, highlevel=True, attrs=OTHER_ATTRS).attrs is OTHER_ATTRS


def test_string_operations_unary_with_arg_slice():
    pyarrow = pytest.importorskip("pyarrow")
    if packaging.version.Version(pyarrow.__version__) < packaging.version.Version("13"):
        pytest.xfail("pyarrow<13 fails to perform this slice")
    # Default no attrs
    assert (
        ak.str.slice(
            [["hello", "world!"], [], ["it's a beautiful day!"]],
            1,
            highlevel=True,
        ).attrs
        == {}
    )
    # Carry from argument
    assert (
        ak.str.slice(
            [["hello", "world!"], [], ["it's a beautiful day!"]],
            1,
            highlevel=True,
            attrs=SOME_ATTRS,
        ).attrs
        is SOME_ATTRS
    )
    # Carry from outer array
    array = ak.Array(
        [["hello", "world!"], [], ["it's a beautiful day!"]], attrs=SOME_ATTRS
    )
    assert ak.str.slice(array, 1, highlevel=True).attrs is SOME_ATTRS
    # Carry from argument exclusively
    assert (
        ak.str.slice(array, 1, highlevel=True, attrs=OTHER_ATTRS).attrs is OTHER_ATTRS
    )


@pytest.mark.parametrize(
    "func",
    [
        # Containment
        ak.str.index_in,
        ak.str.is_in,
        # Splitting and joining
        ak.str.join,
        # This function is 1+ args, but we will test the binary variant
        ak.str.join_element_wise,
    ],
)
def test_string_operations_binary(func):
    pytest.importorskip("pyarrow")
    assert (
        func(
            [["hello", "world!"], [], ["it's a beautiful day!"]],
            ["hello"],
            highlevel=True,
        ).attrs
        == {}
    )
    assert (
        func(
            [["hello", "world!"], [], ["it's a beautiful day!"]],
            ["hello"],
            highlevel=True,
            attrs=SOME_ATTRS,
        ).attrs
        is SOME_ATTRS
    )
    # Carry from first array
    array = ak.Array(
        [["hello", "world!"], [], ["it's a beautiful day!"]], attrs=SOME_ATTRS
    )
    assert func(array, ["hello"], highlevel=True).attrs is SOME_ATTRS

    # Carry from second array
    value_array = ak.Array(["hello"], attrs=OTHER_ATTRS)
    assert (
        func(
            [["hello", "world!"], [], ["it's a beautiful day!"]],
            value_array,
            highlevel=True,
        ).attrs
        is OTHER_ATTRS
    )
    # Carry from both arrays
    assert func(
        array,
        value_array,
        highlevel=True,
    ).attrs == {**OTHER_ATTRS, **SOME_ATTRS}

    # Carry from argument
    assert (
        func(array, value_array, highlevel=True, attrs=OTHER_ATTRS).attrs is OTHER_ATTRS
    )


def test_broadcasting_arrays():
    left = ak.Array([1, 2, 3], attrs=SOME_ATTRS)
    right = ak.Array([1], attrs=OTHER_ATTRS)

    left_result, right_result = ak.broadcast_arrays(left, right)
    assert left_result.attrs is SOME_ATTRS
    assert right_result.attrs is OTHER_ATTRS


def test_broadcasting_fields():
    left = ak.Array([{"x": 1}, {"x": 2}], attrs=SOME_ATTRS)
    right = ak.Array([{"y": 1}, {"y": 2}], attrs=OTHER_ATTRS)

    left_result, right_result = ak.broadcast_fields(left, right)
    assert left_result.attrs is SOME_ATTRS
    assert right_result.attrs is OTHER_ATTRS


def test_numba_arraybuilder():
    numba = pytest.importorskip("numba")
    builder = ak.ArrayBuilder(attrs=SOME_ATTRS)
    assert builder.attrs is SOME_ATTRS

    @numba.njit
    def func(array):
        return array

    assert func(builder).attrs is SOME_ATTRS


def test_numba_array():
    numba = pytest.importorskip("numba")
    array = ak.Array([1, 2, 3], attrs=SOME_ATTRS)
    assert array.attrs is SOME_ATTRS

    @numba.njit
    def func(array):
        return array

    assert func(array).attrs is SOME_ATTRS
