# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.virtual import VirtualNDArray
from awkward.operations.ak_concatenate import enforce_concatenated_form

INCOMPATIBLE = ak.forms.UnionForm(
    "i8",
    "i64",
    [
        ak.forms.NumpyForm("float64"),
        ak.forms.RecordForm([ak.forms.NumpyForm("int64")], ["x"]),
    ],
)


@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
@pytest.mark.parametrize("primitive", ["int64", "float64"])
@pytest.mark.parametrize("reverse", [False, True])
def test_add_union(backend, primitive, reverse):
    layout = ak.to_layout([1, 2, 3]).to_backend(backend)
    number = ak.forms.NumpyForm(primitive)
    string = ak.to_layout(["text"]).form
    forms = [number, string] if not reverse else [string, number]
    target = ak.forms.UnionForm("i8", "i64", forms)

    result = enforce_concatenated_form(layout, target)

    assert result.form == target
    assert result.backend is layout.backend
    if backend == "cpu":
        assert result.to_list() == [1, 2, 3]
        assert result.contents[result.tags[0]].form == number


@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
@pytest.mark.parametrize("grow", [False, True])
def test_preserve_union(backend, grow):
    layout = ak.to_layout([1, "two", 3]).to_backend(backend)
    number = ak.forms.NumpyForm("float64")
    string = layout.contents[1].form
    record = ak.forms.RecordForm([ak.forms.NumpyForm("int64")], ["x"])
    # Reverse the input order to require a nontrivial compatibility permutation.
    forms = [string, number, record] if grow else [string, number]
    target = ak.forms.UnionForm("i8", "i64", forms)

    result = enforce_concatenated_form(layout, target)

    assert result.form == target
    assert result.backend is layout.backend
    if backend == "cpu":
        assert result.to_list() == [1, "two", 3]


@pytest.mark.parametrize("reverse", [False, True])
def test_exact_type_takes_precedence(reverse):
    layout = ak.to_layout([True, False])
    integer = ak.forms.NumpyForm("int64")
    boolean = ak.forms.NumpyForm("bool")
    # Such a union can result from concatenation with mergebool=False.
    forms = [integer, boolean] if not reverse else [boolean, integer]
    target = ak.forms.UnionForm("i8", "i64", forms)

    result = enforce_concatenated_form(layout, target)

    assert result.form == target
    assert result.to_list() == [True, False]
    assert result.contents[result.tags[0]].form == boolean


def test_add_union_incompatible_raises():
    # Previously returned a UnionArray whose index pointed past the end of an
    # empty content (https://github.com/scikit-hep/awkward/issues/4576).
    with pytest.raises(AssertionError, match="mergeable with the layout"):
        enforce_concatenated_form(ak.to_layout(["a"]), INCOMPATIBLE)


def test_preserve_union_incompatible_raises():
    with pytest.raises(AssertionError, match="some permutation"):
        enforce_concatenated_form(ak.to_layout([1, "two"]), INCOMPATIBLE)


def test_only_absent_branches_allocate_empty_layouts(monkeypatch):
    # Compatibility is checked on forms; empty layouts are built only for
    # target branches that the layout does not fill.
    allocated = []
    original = ak.forms.Form.length_zero_array

    def track(form, *args, **kwargs):
        allocated.append(form)
        return original(form, *args, **kwargs)

    monkeypatch.setattr(ak.forms.Form, "length_zero_array", track)

    number = ak.forms.NumpyForm("float64")
    string = ak.to_layout(["text"]).form
    enforce_concatenated_form(
        ak.to_layout([1, 2, 3]), ak.forms.UnionForm("i8", "i64", [string, number])
    )
    assert allocated == [string]


@pytest.mark.parametrize("preserve_union", [False, True])
@pytest.mark.parametrize("promote_option", [False, True])
def test_virtual_data_stays_lazy(preserve_union, promote_option):
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
    assert result.form == target


@pytest.mark.parametrize(
    "arrays",
    [
        [[1, 2], ["a"]],
        [[1, 2], [1.5]],
        [[[1]], [None]],
        [[{"x": 1}], [{"x": 1.5}]],
        [[1, "a"], [{"x": 1}]],
        [[1, "a"], [2.5]],
    ],
    ids=["add-union", "promote", "option", "record", "grow-union", "merge-in-union"],
)
def test_matches_concatenated_typetracer_form(arrays):
    # The contract relied on by dask-awkward: every input, enforced to the form
    # of the concatenated typetracers, has that form and keeps its values.
    arrays = [ak.Array(x) for x in arrays]
    form = ak.concatenate(
        [ak.Array(x.layout.to_typetracer(forget_length=True)) for x in arrays]
    ).layout.form

    for array in arrays:
        result = enforce_concatenated_form(array.layout, form)
        assert result.form.is_equal_to(form, all_parameters=True, form_key=False)
        assert result.to_list() == array.to_list()
