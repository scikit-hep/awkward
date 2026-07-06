# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak


def test_NumpyForm():
    form = ak.forms.NumpyForm("float64")

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * float64"
    assert result.tolist() == [0.0]


def test_NumpyForm_inner_shape():
    form = ak.forms.NumpyForm("float64", inner_shape=(2, 3))

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * 2 * 3 * float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * 2 * 3 * float64"
    assert result.tolist() == [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]


def test_EmptyForm():
    form = ak.forms.EmptyForm()

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * unknown"
    assert result.tolist() == []

    with pytest.raises(TypeError, match="cannot generate a length_one_array"):
        form.length_one_array()


def test_RegularForm():
    form = ak.forms.RegularForm(ak.forms.NumpyForm("float64"), 3)

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * 3 * float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * 3 * float64"
    assert result.tolist() == [[0.0, 0.0, 0.0]]


def test_ListOffsetForm():
    form = ak.forms.ListOffsetForm("i64", ak.forms.NumpyForm("float64"))

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * var * float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * var * float64"
    assert result.tolist() == [[]]


def test_ListForm():
    form = ak.forms.ListForm("i64", "i64", ak.forms.NumpyForm("float64"))

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * var * float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * var * float64"
    assert result.tolist() == [[]]


def test_RecordForm():
    form = ak.forms.RecordForm(
        [ak.forms.NumpyForm("float64"), ak.forms.NumpyForm("int64")],
        ["x", "y"],
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * {x: float64, y: int64}"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * {x: float64, y: int64}"
    assert result.tolist() == [{"x": 0.0, "y": 0}]


def test_IndexedForm():
    form = ak.forms.IndexedForm("i64", ak.forms.NumpyForm("float64"))

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * float64"
    assert result.tolist() == [0.0]


def test_IndexedOptionForm():
    form = ak.forms.IndexedOptionForm("i64", ak.forms.NumpyForm("float64"))

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * ?float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * ?float64"
    assert result.tolist() == [None]


def test_ByteMaskedForm():
    form = ak.forms.ByteMaskedForm("i8", ak.forms.NumpyForm("float64"), valid_when=True)

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * ?float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * ?float64"
    assert result.tolist() == [None]


def test_BitMaskedForm():
    form = ak.forms.BitMaskedForm(
        "u8", ak.forms.NumpyForm("float64"), valid_when=True, lsb_order=True
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * ?float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * ?float64"
    assert result.tolist() == [None]


def test_UnmaskedForm():
    form = ak.forms.UnmaskedForm(ak.forms.NumpyForm("float64"))

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * ?float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * ?float64"
    assert result.tolist() == [0.0]


def test_UnionForm():
    form = ak.forms.UnionForm(
        "i8",
        "i64",
        [ak.forms.NumpyForm("float64"), ak.forms.NumpyForm("int64")],
    )

    result = ak.Array(form.length_zero_array())
    assert len(result) == 0

    result = ak.Array(form.length_one_array())
    assert len(result) == 1
    assert result.tolist() == [0.0]


def test_RegularForm_IndexedOptionForm():
    form = ak.forms.RegularForm(
        ak.forms.IndexedOptionForm("i64", ak.forms.NumpyForm("float64")), 10
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * 10 * ?float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * 10 * ?float64"
    assert result.tolist() == [[None] * 10]


def test_RegularForm_ByteMaskedForm():
    form = ak.forms.RegularForm(
        ak.forms.ByteMaskedForm("i8", ak.forms.NumpyForm("float64"), valid_when=True), 5
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * 5 * ?float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * 5 * ?float64"
    assert result.tolist() == [[None] * 5]


def test_RegularForm_BitMaskedForm():
    form = ak.forms.RegularForm(
        ak.forms.BitMaskedForm(
            "u8", ak.forms.NumpyForm("float64"), valid_when=True, lsb_order=True
        ),
        8,
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * 8 * ?float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * 8 * ?float64"
    assert result.tolist() == [[None] * 8]


def test_RegularForm_IndexedForm():
    form = ak.forms.RegularForm(
        ak.forms.IndexedForm("i64", ak.forms.NumpyForm("float64")), 7
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * 7 * float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * 7 * float64"
    assert result.tolist() == [[0.0] * 7]


def test_RegularForm_ListOffsetForm():
    form = ak.forms.RegularForm(
        ak.forms.ListOffsetForm("i64", ak.forms.NumpyForm("float64")), 4
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * 4 * var * float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * 4 * var * float64"
    assert result.tolist() == [[[], [], [], []]]


def test_RegularForm_UnionForm():
    form = ak.forms.RegularForm(
        ak.forms.UnionForm(
            "i8",
            "i64",
            [ak.forms.NumpyForm("float64"), ak.forms.NumpyForm("int64")],
        ),
        3,
    )

    result = ak.Array(form.length_zero_array())
    assert len(result) == 0

    result = ak.Array(form.length_one_array())
    assert len(result) == 1
    assert result.tolist() == [[0.0, 0.0, 0.0]]


def test_RegularForm_RecordForm():
    form = ak.forms.RegularForm(
        ak.forms.RecordForm(
            [ak.forms.NumpyForm("float64"), ak.forms.NumpyForm("int64")],
            ["x", "y"],
        ),
        2,
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * 2 * {x: float64, y: int64}"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * 2 * {x: float64, y: int64}"
    assert result.tolist() == [[{"x": 0.0, "y": 0}, {"x": 0.0, "y": 0}]]


def test_RegularForm_RegularForm_IndexedOptionForm():
    form = ak.forms.RegularForm(
        ak.forms.RegularForm(
            ak.forms.IndexedOptionForm("i64", ak.forms.NumpyForm("float64")), 5
        ),
        3,
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * 3 * 5 * ?float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * 3 * 5 * ?float64"
    assert result.tolist() == [[[None] * 5] * 3]


def test_IndexedOptionForm_EmptyForm():
    form = ak.forms.IndexedOptionForm("i64", ak.forms.EmptyForm())

    result = ak.Array(form.length_zero_array())
    assert len(result) == 0

    result = ak.Array(form.length_one_array())
    assert len(result) == 1
    assert result.tolist() == [None]


def test_RegularForm_IndexedOptionForm_EmptyForm():
    form = ak.forms.RegularForm(
        ak.forms.IndexedOptionForm("i64", ak.forms.EmptyForm()), 5
    )

    result = ak.Array(form.length_zero_array())
    assert len(result) == 0

    result = ak.Array(form.length_one_array())
    assert len(result) == 1
    assert result.tolist() == [[None] * 5]


def test_ByteMaskedForm_EmptyForm():
    form = ak.forms.ByteMaskedForm("i8", ak.forms.EmptyForm(), valid_when=True)

    result = ak.Array(form.length_zero_array())
    assert len(result) == 0

    # ByteMaskedForm cannot hide EmptyForm - content length must equal mask length
    with pytest.raises(TypeError, match="cannot generate a length_one_array"):
        form.length_one_array()


def test_BitMaskedForm_EmptyForm():
    form = ak.forms.BitMaskedForm(
        "u8", ak.forms.EmptyForm(), valid_when=True, lsb_order=True
    )

    result = ak.Array(form.length_zero_array())
    assert len(result) == 0

    # BitMaskedForm cannot hide EmptyForm - content length must equal mask length
    with pytest.raises(TypeError, match="cannot generate a length_one_array"):
        form.length_one_array()


def test_RegularForm_ListForm():
    form = ak.forms.RegularForm(
        ak.forms.ListForm("i64", "i64", ak.forms.NumpyForm("float64")), 3
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * 3 * var * float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * 3 * var * float64"
    assert result.tolist() == [[[], [], []]]


def test_RecordForm_tuple():
    form = ak.forms.RecordForm(
        [ak.forms.NumpyForm("float64"), ak.forms.NumpyForm("int64")],
        None,
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * (float64, int64)"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * (float64, int64)"
    assert result.tolist() == [(0.0, 0)]


def test_RegularForm_BitMaskedForm_large():
    # Test bit packing across byte boundaries (size > 8)
    form = ak.forms.RegularForm(
        ak.forms.BitMaskedForm(
            "u8", ak.forms.NumpyForm("float64"), valid_when=True, lsb_order=True
        ),
        10,
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * 10 * ?float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * 10 * ?float64"
    assert result.tolist() == [[None] * 10]


def test_RegularForm_UnmaskedForm():
    form = ak.forms.RegularForm(ak.forms.UnmaskedForm(ak.forms.NumpyForm("float64")), 4)

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * 4 * ?float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * 4 * ?float64"
    assert result.tolist() == [[0.0, 0.0, 0.0, 0.0]]


def test_ByteMaskedForm_valid_when_false():
    form = ak.forms.ByteMaskedForm(
        "i8", ak.forms.NumpyForm("float64"), valid_when=False
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * ?float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * ?float64"
    assert result.tolist() == [None]


def test_BitMaskedForm_valid_when_false():
    form = ak.forms.BitMaskedForm(
        "u8", ak.forms.NumpyForm("float64"), valid_when=False, lsb_order=True
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * ?float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * ?float64"
    assert result.tolist() == [None]


def test_BitMaskedForm_ListOffsetForm():
    form = ak.forms.BitMaskedForm(
        "u8",
        ak.forms.ListOffsetForm("i64", ak.forms.NumpyForm("float64")),
        valid_when=True,
        lsb_order=True,
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * option[var * float64]"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * option[var * float64]"
    assert result.tolist() == [None]


def test_ByteMaskedForm_ListOffsetForm():
    form = ak.forms.ByteMaskedForm(
        "i8",
        ak.forms.ListOffsetForm("i64", ak.forms.NumpyForm("float64")),
        valid_when=True,
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * option[var * float64]"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * option[var * float64]"
    assert result.tolist() == [None]


def test_BitMaskedForm_ListForm():
    form = ak.forms.BitMaskedForm(
        "u8",
        ak.forms.ListForm("i64", "i64", ak.forms.NumpyForm("float64")),
        valid_when=True,
        lsb_order=True,
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * option[var * float64]"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * option[var * float64]"
    assert result.tolist() == [None]


def test_ByteMaskedForm_ListForm():
    form = ak.forms.ByteMaskedForm(
        "i8",
        ak.forms.ListForm("i64", "i64", ak.forms.NumpyForm("float64")),
        valid_when=True,
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * option[var * float64]"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * option[var * float64]"
    assert result.tolist() == [None]


def test_IndexedOptionForm_ListOffsetForm():
    form = ak.forms.IndexedOptionForm(
        "i64", ak.forms.ListOffsetForm("i64", ak.forms.NumpyForm("float64"))
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * option[var * float64]"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * option[var * float64]"
    assert result.tolist() == [None]


def test_IndexedOptionForm_ListForm():
    form = ak.forms.IndexedOptionForm(
        "i64", ak.forms.ListForm("i64", "i64", ak.forms.NumpyForm("float64"))
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * option[var * float64]"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * option[var * float64]"
    assert result.tolist() == [None]


def test_ListOffsetForm_BitMaskedForm():
    form = ak.forms.ListOffsetForm(
        "i64",
        ak.forms.BitMaskedForm(
            "u8", ak.forms.NumpyForm("float64"), valid_when=True, lsb_order=True
        ),
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * var * ?float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * var * ?float64"
    assert result.tolist() == [[]]


def test_ListOffsetForm_ByteMaskedForm():
    form = ak.forms.ListOffsetForm(
        "i64",
        ak.forms.ByteMaskedForm("i8", ak.forms.NumpyForm("float64"), valid_when=True),
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * var * ?float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * var * ?float64"
    assert result.tolist() == [[]]


def test_ListOffsetForm_IndexedOptionForm():
    form = ak.forms.ListOffsetForm(
        "i64",
        ak.forms.IndexedOptionForm("i64", ak.forms.NumpyForm("float64")),
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * var * ?float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * var * ?float64"
    assert result.tolist() == [[]]


def test_ListForm_BitMaskedForm():
    form = ak.forms.ListForm(
        "i64",
        "i64",
        ak.forms.BitMaskedForm(
            "u8", ak.forms.NumpyForm("float64"), valid_when=True, lsb_order=True
        ),
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * var * ?float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * var * ?float64"
    assert result.tolist() == [[]]


def test_ListForm_ByteMaskedForm():
    form = ak.forms.ListForm(
        "i64",
        "i64",
        ak.forms.ByteMaskedForm("i8", ak.forms.NumpyForm("float64"), valid_when=True),
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * var * ?float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * var * ?float64"
    assert result.tolist() == [[]]


def test_ListForm_IndexedOptionForm():
    form = ak.forms.ListForm(
        "i64",
        "i64",
        ak.forms.IndexedOptionForm("i64", ak.forms.NumpyForm("float64")),
    )

    result = ak.Array(form.length_zero_array())
    assert str(result.type) == "0 * var * ?float64"
    assert result.tolist() == []

    result = ak.Array(form.length_one_array())
    assert str(result.type) == "1 * var * ?float64"
    assert result.tolist() == [[]]
