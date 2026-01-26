# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test_flat_arrays():
    array = ak.Array(ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype="f8")))
    form, length, container = ak.to_buffers(array, byteorder="<")
    assert form.is_equal_to(ak.forms.NumpyForm("float64"))
    assert length == 3
    assert container.keys() == {"node0-data"}
    np.testing.assert_array_equal(
        container["node0-data"],
        ak._util.native_to_byteorder(np.array([1.0, 2.0, 3.0], dtype="f8"), "<"),
    )
    reconstructed = ak.from_buffers(form, length, container, byteorder="<")
    assert ak.array_equal(array, reconstructed)

    array = ak.Array(ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype="f8")))
    form, length, container = ak.to_buffers(array, byteorder=">")
    assert form.is_equal_to(ak.forms.NumpyForm("float64"))
    assert length == 3
    assert container.keys() == {"node0-data"}
    np.testing.assert_array_equal(
        container["node0-data"],
        ak._util.native_to_byteorder(np.array([1.0, 2.0, 3.0], dtype="f8"), ">"),
    )
    reconstructed = ak.from_buffers(form, length, container, byteorder=">")
    assert ak.array_equal(array, reconstructed)


def test_jagged_arrays():
    array = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index(np.array([0, 2, 2, 3], dtype="i8")),
            ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype="f8")),
        )
    )
    form, length, container = ak.to_buffers(array, byteorder="<")
    assert form.is_equal_to(
        ak.forms.ListOffsetForm("i64", ak.forms.NumpyForm("float64"))
    )
    assert length == 3
    assert container.keys() == {"node0-offsets", "node1-data"}
    np.testing.assert_array_equal(
        container["node0-offsets"],
        ak._util.native_to_byteorder(np.array([0, 2, 2, 3], dtype="i8"), "<"),
    )
    np.testing.assert_array_equal(
        container["node1-data"],
        ak._util.native_to_byteorder(np.array([1.0, 2.0, 3.0], dtype="f8"), "<"),
    )
    reconstructed = ak.from_buffers(form, length, container, byteorder="<")
    assert ak.array_equal(array, reconstructed)

    array = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index(np.array([0, 2, 2, 3], dtype="i8")),
            ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype="f8")),
        )
    )
    form, length, container = ak.to_buffers(array, byteorder=">")
    assert form.is_equal_to(
        ak.forms.ListOffsetForm("i64", ak.forms.NumpyForm("float64"))
    )
    assert length == 3
    assert container.keys() == {"node0-offsets", "node1-data"}
    np.testing.assert_array_equal(
        container["node0-offsets"],
        ak._util.native_to_byteorder(np.array([0, 2, 2, 3], dtype="i8"), ">"),
    )
    np.testing.assert_array_equal(
        container["node1-data"],
        ak._util.native_to_byteorder(np.array([1.0, 2.0, 3.0], dtype="f8"), ">"),
    )
    reconstructed = ak.from_buffers(form, length, container, byteorder=">")
    assert ak.array_equal(array, reconstructed)


def test_flat_bytes():
    form = ak.forms.NumpyForm("float64", form_key="node0")
    length = 3
    container = {"node0-data": np.array([1.0, 2.0, 3.0], dtype="<f8").tobytes()}
    array = ak.from_buffers(form, length, container, byteorder="<")
    assert ak.array_equal(
        array, ak.Array(ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype="f8")))
    )

    form = ak.forms.NumpyForm("float64", form_key="node0")
    length = 3
    container = {"node0-data": np.array([1.0, 2.0, 3.0], dtype="<f8").tobytes()}
    array = ak.from_buffers(form, length, container, byteorder=">")
    assert ak.array_equal(
        array,
        ak.Array(
            ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype="f8").byteswap())
        ),
    )

    form = ak.forms.NumpyForm("float64", form_key="node0")
    length = 3
    container = {"node0-data": np.array([1.0, 2.0, 3.0], dtype=">f8").tobytes()}
    array = ak.from_buffers(form, length, container, byteorder="<")
    assert ak.array_equal(
        array,
        ak.Array(
            ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype="f8").byteswap())
        ),
    )

    form = ak.forms.NumpyForm("float64", form_key="node0")
    length = 3
    container = {"node0-data": np.array([1.0, 2.0, 3.0], dtype=">f8").tobytes()}
    array = ak.from_buffers(form, length, container, byteorder=">")
    assert ak.array_equal(
        array, ak.Array(ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype="f8")))
    )


def test_jagged_bytes():
    form = ak.forms.ListOffsetForm(
        "i64", ak.forms.NumpyForm("float64", form_key="node1"), form_key="node0"
    )
    length = 3
    container = {
        "node0-offsets": np.array([0, 2, 2, 3], dtype="<i8").tobytes(),
        "node1-data": np.array([1.0, 2.0, 3.0], dtype="<f8").tobytes(),
    }
    array = ak.from_buffers(form, length, container, byteorder="<")
    assert ak.array_equal(
        array,
        ak.Array(
            ak.contents.ListOffsetArray(
                ak.index.Index(np.array([0, 2, 2, 3], dtype="i8")),
                ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype="f8")),
            )
        ),
    )

    form = ak.forms.ListOffsetForm(
        "i64", ak.forms.NumpyForm("float64", form_key="node1"), form_key="node0"
    )
    length = 3
    container = {
        "node0-offsets": np.array([0, 2, 2, 3], dtype="<i8").tobytes(),
        "node1-data": np.array([1.0, 2.0, 3.0], dtype="<f8").tobytes(),
    }
    with pytest.raises((ValueError, OverflowError)):
        array = ak.from_buffers(form, length, container, byteorder=">")

    form = ak.forms.ListOffsetForm(
        "i64", ak.forms.NumpyForm("float64", form_key="node1"), form_key="node0"
    )
    length = 3
    container = {
        "node0-offsets": np.array([0, 2, 2, 3], dtype=">i8").tobytes(),
        "node1-data": np.array([1.0, 2.0, 3.0], dtype=">f8").tobytes(),
    }
    with pytest.raises((ValueError, OverflowError)):
        array = ak.from_buffers(form, length, container, byteorder="<")

    form = ak.forms.ListOffsetForm(
        "i64", ak.forms.NumpyForm("float64", form_key="node1"), form_key="node0"
    )
    length = 3
    container = {
        "node0-offsets": np.array([0, 2, 2, 3], dtype=">i8").tobytes(),
        "node1-data": np.array([1.0, 2.0, 3.0], dtype=">f8").tobytes(),
    }
    array = ak.from_buffers(form, length, container, byteorder=">")
    assert ak.array_equal(
        array,
        ak.Array(
            ak.contents.ListOffsetArray(
                ak.index.Index(np.array([0, 2, 2, 3], dtype="i8")),
                ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype="f8")),
            )
        ),
    )
