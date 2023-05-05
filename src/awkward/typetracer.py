from __future__ import annotations

__all__ = [
    "is_unknown_array",
    "is_unknown_scalar",
    "empty_if_typetracer",
    "typetracer_with_report",
]

import awkward as ak
from awkward._backends.typetracer import TypeTracerBackend
from awkward._behavior import behavior_of
from awkward._layout import wrap_layout
from awkward._nplikes.typetracer import (
    TypeTracerReport,
    is_unknown_array,
    is_unknown_scalar,
)
from awkward._typing import TypeVar
from awkward.highlevel import Array, Record
from awkward.operations.ak_to_layout import to_layout

T = TypeVar("T", Array, Record)


def empty_if_typetracer(array: T) -> T:
    typetracer_backend = TypeTracerBackend.instance()

    layout = to_layout(array, allow_other=False)
    behavior = behavior_of(array)

    if layout.backend is typetracer_backend:
        layout._touch_data(True)
        return layout.form.length_zero_array(behavior=behavior)
    else:
        return wrap_layout(layout, behavior=behavior)


def _attach_report(
    layout: ak.contents.Content, form: ak.forms.Form, report: TypeTracerReport
):
    if isinstance(layout, (ak.contents.BitMaskedArray, ak.contents.ByteMaskedArray)):
        assert isinstance(form, (ak.forms.BitMaskedForm, ak.forms.ByteMaskedForm))
        layout.mask.data.form_key = form.form_key
        layout.mask.data.report = report
        _attach_report(layout.content, form.content, report)

    elif isinstance(layout, ak.contents.EmptyArray):
        assert isinstance(form, ak.forms.EmptyForm)

    elif isinstance(layout, (ak.contents.IndexedArray, ak.contents.IndexedOptionArray)):
        assert isinstance(form, (ak.forms.IndexedForm, ak.forms.IndexedOptionForm))
        layout.index.data.form_key = form.form_key
        layout.index.data.report = report
        _attach_report(layout.content, form.content, report)

    elif isinstance(layout, ak.contents.ListArray):
        assert isinstance(form, ak.forms.ListForm)
        layout.starts.data.form_key = form.form_key
        layout.starts.data.report = report
        layout.stops.data.form_key = form.form_key
        layout.stops.data.report = report
        _attach_report(layout.content, form.content, report)

    elif isinstance(layout, ak.contents.ListOffsetArray):
        assert isinstance(form, ak.forms.ListOffsetForm)
        layout.offsets.data.form_key = form.form_key
        layout.offsets.data.report = report
        _attach_report(layout.content, form.content, report)

    elif isinstance(layout, ak.contents.NumpyArray):
        assert isinstance(form, ak.forms.NumpyForm)
        layout.data.form_key = form.form_key
        layout.data.report = report

    elif isinstance(layout, ak.contents.RecordArray):
        assert isinstance(form, ak.forms.RecordForm)
        for x, y in zip(layout.contents, form.contents):
            _attach_report(x, y, report)

    elif isinstance(layout, (ak.contents.RegularArray, ak.contents.UnmaskedArray)):
        assert isinstance(form, (ak.forms.RegularForm, ak.forms.UnmaskedForm))
        _attach_report(layout.content, form.content, report)

    elif isinstance(layout, ak.contents.UnionArray):
        assert isinstance(form, ak.forms.UnionForm)
        layout.tags.data.form_key = form.form_key
        layout.tags.data.report = report
        layout.index.data.form_key = form.form_key
        layout.index.data.report = report
        for x, y in zip(layout.contents, form.contents):
            _attach_report(x, y, report)

    else:
        raise AssertionError(f"unrecognized layout type {type(layout)}")


def typetracer_with_report(
    form: ak.forms.Form, forget_length: bool = True
) -> tuple[ak.contents.Content, TypeTracerReport]:
    layout = form.length_zero_array(highlevel=False).to_typetracer(
        forget_length=forget_length
    )
    report = TypeTracerReport()
    _attach_report(layout, form, report)
    return layout, report
