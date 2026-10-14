# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import itertools

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.virtual import VirtualNDArray


def representative_forms():
    f = ak.forms
    integer = f.NumpyForm("int64")
    floating = f.NumpyForm("float64")
    forms = [
        f.EmptyForm(),
        *[
            f.NumpyForm(t)
            for t in (
                "bool",
                "int8",
                "uint64",
                "float32",
                "float64",
                "complex128",
                "datetime64[ns]",
                "datetime64[ms]",
                "timedelta64[ns]",
            )
        ],
        integer,
        f.NumpyForm("int64", inner_shape=(2, 3)),
        f.NumpyForm("int64", inner_shape=(0,)),
        f.RegularForm(integer, 2),
        f.RegularForm(integer, 3),
        f.ListForm("i64", "i64", integer),
        f.ListOffsetForm("i64", integer),
        f.RecordForm([integer, floating], ["x", "y"]),
        f.RecordForm([floating, integer], ["y", "x"]),
        f.RecordForm([integer], ["z"]),
        f.RecordForm([integer, floating], None),
        f.RecordForm([], []),
        f.RecordForm([], None),
        f.UnionForm("i8", "i64", [integer, f.ListOffsetForm("i64", integer)]),
        f.ListOffsetForm(
            "i64",
            f.NumpyForm("uint8", parameters={"__array__": "char"}),
            parameters={"__array__": "string"},
        ),
        f.RecordForm([integer], ["x"], parameters={"__record__": "A"}),
        f.RecordForm([integer], ["x"], parameters={"__record__": "B"}),
        f.NumpyForm("int64", parameters={"custom": "ignored"}, form_key="data"),
    ]
    for content in (integer, f.EmptyForm(), forms[17]):
        forms.extend(
            [
                f.IndexedForm("i64", content),
                f.IndexedOptionForm("i64", content),
                f.ByteMaskedForm("i8", content, True),
                f.BitMaskedForm("u8", content, True, True),
                f.UnmaskedForm(content),
            ]
        )
    return forms


@pytest.mark.parametrize("mergebool", [True, False])
@pytest.mark.parametrize("mergecastable", ["same_kind", "equiv", "family"])
def test_form_content_typetracer_parity(mergebool, mergecastable):
    forms = representative_forms()
    layouts = [form.length_zero_array(highlevel=False) for form in forms]
    tracers = [layout.to_typetracer(forget_length=True) for layout in layouts]
    for i, j in itertools.product(range(len(forms)), repeat=2):
        # Existing Content predicates can fall through to None for unequal
        # tuple arities; compare their boolean compatibility decision.
        expected = bool(
            ak._do.mergeable(layouts[i], layouts[j], mergebool, mergecastable)
        )
        assert (
            ak._do.mergeable(forms[i], forms[j], mergebool, mergecastable) == expected
        ), (forms[i], forms[j], mergebool, mergecastable)
        assert (
            bool(ak._do.mergeable(tracers[i], tracers[j], mergebool, mergecastable))
            == expected
        )
        assert (
            ak._do.mergeable(forms[i], layouts[j], mergebool, mergecastable) == expected
        )
        assert (
            ak._do.mergeable(layouts[i], forms[j], mergebool, mergecastable) == expected
        )


@pytest.mark.parametrize(
    "policy,expected", [("same_kind", True), ("equiv", False), ("family", False)]
)
def test_casting_policy(policy, expected):
    assert (
        ak._do.mergeable(
            ak.forms.NumpyForm("int64"),
            ak.forms.NumpyForm("float64"),
            mergecastable=policy,
        )
        is expected
    )


def test_metadata_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("mergeability must not construct buffers or materialize data")

    layout = ak.contents.NumpyArray(
        VirtualNDArray(
            Numpy.instance(),
            shape=(10, 2, 3),
            dtype=np.dtype("int64"),
            generator=forbidden,
        )
    )
    form = ak.forms.NumpyForm("float64", inner_shape=(2, 3))
    monkeypatch.setattr(ak.forms.Form, "length_zero_array", forbidden)
    monkeypatch.setattr(ak.forms.Form, "length_one_array", forbidden)
    monkeypatch.setattr(ak, "from_buffers", forbidden)
    assert ak._do.mergeable(layout, form)
    assert ak._do.mergeable(form, layout)
    assert ak._do.mergeable(layout.form, form)
    assert not ak._do.mergeable(layout, form, mergecastable="family")


@pytest.mark.parametrize("mergecastable", ["same_kind", "equiv", "family"])
def test_nonempty_layouts(mergecastable):
    layouts = [
        ak.to_layout(data)
        for data in (
            [1, 2],
            [1.5, 2.5],
            [True, False],
            [[1], [], [2, 3]],
            [None, 1],
            [{"x": 1, "y": [2]}],
            [(1, 2)],
            ["one", "two"],
            [1, "two"],
        )
    ]
    layouts.append(ak.contents.NumpyArray(np.arange(12).reshape(2, 2, 3)))
    for one, two in itertools.product(layouts, repeat=2):
        for mergebool in (True, False):
            assert ak._do.mergeable(
                one.form, two.form, mergebool, mergecastable
            ) == bool(ak._do.mergeable(one, two, mergebool, mergecastable))


def test_invalid_casting_policy():
    with pytest.raises(TypeError, match="unrecognized mergecastable"):
        ak._do.mergeable(
            ak.forms.NumpyForm("int32"),
            ak.forms.NumpyForm("int64"),
            mergecastable="invalid",
        )
