# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.virtual import VirtualNDArray
from awkward.operations.ak_concatenate import enforce_concatenated_form


@pytest.fixture
def empty_layouts(monkeypatch):
    allocated = []
    original = ak.forms.Form.length_zero_array

    def track(form, *args, **kwargs):
        allocated.append(form)
        return original(form, *args, **kwargs)

    monkeypatch.setattr(ak.forms.Form, "length_zero_array", track)
    return allocated


@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
@pytest.mark.parametrize("primitive", ["int64", "float64"])
@pytest.mark.parametrize("reverse", [False, True])
def test_add_union(backend, primitive, reverse, empty_layouts):
    layout = ak.to_layout([1, 2, 3]).to_backend(backend)
    number = ak.forms.NumpyForm(primitive)
    string = ak.to_layout(["text"]).form
    forms = [number, string] if not reverse else [string, number]
    target = ak.forms.UnionForm("i8", "i64", forms)

    result = enforce_concatenated_form(layout, target)

    # Only the absent output branch needs an empty layout. Both exact-type
    # matching and the int-to-float compatibility fallback use metadata.
    assert empty_layouts == [string]
    assert result.contents[0].form == number
    assert result.contents[1].form == string
    assert result.backend is layout.backend
    if backend == "cpu":
        assert result.to_list() == [1, 2, 3]


@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
@pytest.mark.parametrize("grow", [False, True])
def test_preserve_union(backend, grow, empty_layouts):
    layout = ak.to_layout([1, "two", 3]).to_backend(backend)
    number = ak.forms.NumpyForm("float64")
    string = layout.contents[1].form
    record = ak.forms.RecordForm([ak.forms.NumpyForm("int64")], ["x"])
    # Reverse the input order to require a nontrivial compatibility permutation.
    forms = [string, number, record] if grow else [string, number]
    target = ak.forms.UnionForm("i8", "i64", forms)

    result = enforce_concatenated_form(layout, target)

    assert empty_layouts == ([record] if grow else [])
    assert result.contents[0].form == number
    assert result.contents[1].form == string
    assert result.backend is layout.backend
    if grow:
        assert result.contents[2].form == record
    if backend == "cpu":
        assert result.to_list() == [1, "two", 3]
        np.testing.assert_array_equal(result.tags.data, layout.tags.data)
        np.testing.assert_array_equal(result.index.data, layout.index.data)


def test_exact_type_takes_precedence(empty_layouts):
    layout = ak.to_layout([True, False])
    integer = ak.forms.NumpyForm("int64")
    boolean = ak.forms.NumpyForm("bool")
    # Such a union can result from concatenation with mergebool=False.
    target = ak.forms.UnionForm("i8", "i64", [integer, boolean])

    result = enforce_concatenated_form(layout, target)

    assert empty_layouts == [integer]
    assert result.contents[0].form == boolean
    assert result.to_list() == [True, False]


def test_incompatible_union_does_not_allocate(empty_layouts):
    layout = ak.to_layout([1, "two"])
    target = ak.forms.UnionForm(
        "i8",
        "i64",
        [
            ak.forms.NumpyForm("float64"),
            ak.forms.RecordForm([ak.forms.NumpyForm("int64")], ["x"]),
        ],
    )
    with pytest.raises(AssertionError, match="some permutation"):
        enforce_concatenated_form(layout, target)
    assert not empty_layouts


@pytest.mark.parametrize("preserve_union", [False, True])
@pytest.mark.parametrize("promote_option", [False, True])
def test_virtual_data_stays_lazy(preserve_union, promote_option, empty_layouts):
    def forbidden():
        pytest.fail("compatibility matching must not materialize virtual data")

    buffer = VirtualNDArray(
        Numpy.instance(),
        shape=(3,),
        dtype=np.dtype("int64"),
        generator=forbidden,
    )
    numeric = ak.contents.NumpyArray(buffer)
    record = ak.contents.RecordArray([numeric], ["x"])
    numeric_form = numeric.form
    record_form = record.form
    if promote_option:
        numeric_form = ak.forms.UnmaskedForm(numeric_form)
        record_form = ak.forms.UnmaskedForm(record_form)
    target = ak.forms.UnionForm("i8", "i64", [record_form, numeric_form])
    if preserve_union:
        layout = ak.contents.UnionArray(
            ak.index.Index8([0, 1]), ak.index.Index64([0, 0]), [numeric, record]
        )
    else:
        layout = numeric

    result = enforce_concatenated_form(layout, target)

    assert not buffer.is_materialized
    assert result.contents[0].form == numeric_form
    assert empty_layouts == ([] if preserve_union else [record_form])
