# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401


numpy = ak.nplike.Numpy.instance()


@pytest.mark.skip("string broadcasting is broken")
def test_broadcast_strings_1d():
    this = ak._v2.Array(["one", "two", "one", "nine"])
    that = ak._v2.with_parameter(
        ak._v2.Array(["two", "one", "four", "three"]), "reason", "because!"
    )
    this_next, that_next = ak._v2.operations.ak_broadcast_arrays.broadcast_arrays(
        this, that
    )

    assert this.layout.parameters == this_next.layout.parameters
    assert that.layout.parameters == that_next.layout.parameters


@pytest.mark.skip("string broadcasting is broken")
def test_broadcast_strings_1d_right_broadcast():
    this = ak._v2.Array(["one", "two", "one", "nine"])
    that = ak._v2.operations.ak_to_regular.to_regular(
        ak._v2.with_parameter(ak._v2.Array(["two"]), "reason", "because!"), axis=1
    )
    this_next, that_next = ak._v2.operations.ak_broadcast_arrays.broadcast_arrays(
        this, that
    )

    assert this.layout.parameters == this_next.layout.parameters
    assert that.layout.parameters == that_next.layout.parameters


@pytest.mark.skip("string broadcasting is broken")
def test_broadcast_strings_2d():
    this = ak._v2.Array([["one", "two", "one"], ["nine"]])
    that = ak._v2.to_regular(
        ak._v2.with_parameter(ak._v2.Array([["two"], ["three"]]), "reason", "because!"),
        axis=1,
    )
    this_next, that_next = ak._v2.operations.ak_broadcast_arrays.broadcast_arrays(
        this, that
    )

    assert this.layout.parameters == this_next.layout.parameters
    assert that.layout.parameters == that_next.layout.parameters

    assert this.layout.content.parameters == this_next.layout.content.parameters
    assert that.layout.content.parameters == that_next.layout.content.parameters


@pytest.mark.skip("string broadcasting is broken")
def test_broadcast_string_int():
    this = ak._v2.Array(["one", "two", "one", "nine"])
    that = ak._v2.operations.ak_with_parameter.with_parameter(
        ak._v2.Array([1, 2, 1, 9]), "kind", "integer"
    )
    this_next, that_next = ak._v2.operations.ak_broadcast_arrays.broadcast_arrays(
        this, that
    )

    assert this.layout.parameters == this_next.layout.parameters
    assert that.layout.parameters == that_next.layout.parameters


def test_broadcast_float_int():
    this = ak._v2.operations.ak_with_parameter.with_parameter(
        ak._v2.Array([1.0, 2.0, 3.0, 4.0]), "name", "this"
    )
    that = ak._v2.operations.ak_with_parameter.with_parameter(
        ak._v2.Array([1, 2, 1, 9]), "name", "that"
    )
    this_next, that_next = ak._v2.operations.ak_broadcast_arrays.broadcast_arrays(
        this, that
    )

    assert this.layout.parameters == this_next.layout.parameters
    assert that.layout.parameters == that_next.layout.parameters


def test_broadcast_float_int_option():
    this = ak._v2.contents.NumpyArray(numpy.arange(4), parameters={"name": "this"})
    that = ak._v2.contents.ByteMaskedArray(
        ak._v2.index.Index8(numpy.array([0, 1, 0, 1])),
        ak._v2.contents.NumpyArray(
            numpy.arange(4),
        ),
        valid_when=True,
        parameters={"name": "that"},
    )
    this_next, that_next = ak._v2.operations.ak_broadcast_arrays.broadcast_arrays(
        this, that, highlevel=False
    )

    assert this.parameters == this_next.parameters
    assert that.parameters == that_next.parameters


def test_broadcast_float_int_union():
    this = ak._v2.contents.NumpyArray(numpy.arange(4), parameters={"name": "this"})
    that_1 = ak._v2.contents.ByteMaskedArray(
        ak._v2.index.Index8(numpy.array([0, 1, 0, 1], dtype="int8")),
        ak._v2.contents.NumpyArray(
            numpy.arange(4),
        ),
        valid_when=True,
        parameters={"name": "that"},
    )
    that_2 = ak._v2.contents.ByteMaskedArray(
        ak._v2.index.Index8(numpy.array([0, 1, 0, 1], dtype="int8")),
        ak._v2.contents.NumpyArray(
            numpy.arange(4, dtype="complex"),
        ),
        valid_when=True,
        parameters={"name": "other"},
    )
    that = ak._v2.contents.UnionArray(
        ak._v2.index.Index8(numpy.array([0, 1, 0, 1], dtype="int8")),
        ak._v2.index.Index32(numpy.array([0, 0, 1, 1], dtype="int32")),
        [that_1, that_2],
    )
    this_next, that_next = ak._v2.operations.ak_broadcast_arrays.broadcast_arrays(
        this, that, highlevel=False
    )

    assert this.parameters == this_next.parameters
    assert that.parameters == that_next.parameters


def test_broadcast_float_int_2d():
    this = ak._v2.operations.ak_with_parameter.with_parameter(
        ak._v2.Array([[1.0, 2.0, 3.0], [4.0]]), "name", "this"
    )
    that = ak._v2.operations.ak_with_parameter.with_parameter(
        ak._v2.Array([[1, 2, 1], [9]]), "name", "that"
    )
    this_next, that_next = ak._v2.operations.ak_broadcast_arrays.broadcast_arrays(
        this, that
    )

    assert this.layout.parameters == this_next.layout.parameters
    assert that.layout.parameters == that_next.layout.parameters

    assert this.layout.content.parameters == this_next.layout.content.parameters
    assert that.layout.content.parameters == that_next.layout.content.parameters


def test_broadcast_float_int_2d_right_broadcast():
    this = ak._v2.operations.ak_with_parameter.with_parameter(
        ak._v2.Array([[1.0, 2.0, 3.0], [4.0]]), "name", "this"
    )
    that = ak._v2.operations.ak_to_regular.to_regular(
        ak._v2.operations.ak_with_parameter.with_parameter(
            ak._v2.Array([[1], [9]]), "name", "that"
        ),
        axis=1,
    )
    this_next, that_next = ak._v2.operations.ak_broadcast_arrays.broadcast_arrays(
        this, that
    )

    assert this.layout.parameters == this_next.layout.parameters
    assert that.layout.parameters == that_next.layout.parameters

    assert this.layout.content.parameters == this_next.layout.content.parameters
    assert that.layout.content.parameters == that_next.layout.content.parameters
