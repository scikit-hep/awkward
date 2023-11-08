# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak
from awkward.typetracer import typetracer_with_report

SOME_ATTRS = {"foo": "FOO"}


def test_typetracer_with_report():
    array = ak.zip(
        {
            "x": [[0.2, 0.3, 0.4], [1, 2, 3], [1, 1, 2]],
            "y": [[0.1, 0.1, 0.2], [3, 1, 2], [2, 1, 2]],
            "z": [[0.1, 0.1, 0.2], [3, 1, 2], [2, 1, 2]],
        }
    )
    layout = ak.to_layout(array)
    form = layout.form_with_key("node{id}")

    meta, report = typetracer_with_report(form, highlevel=True, attrs=SOME_ATTRS)
    assert meta.attrs is SOME_ATTRS

    meta, report = typetracer_with_report(form, highlevel=True, attrs=None)
    assert meta._attrs is None


@pytest.mark.parametrize(
    "function",
    [
        ak.typetracer.touch_data,
        ak.typetracer.length_zero_if_typetracer,
        ak.typetracer.length_one_if_typetracer,
    ],
)
def test_function(function):
    array = ak.zip(
        {
            "x": [[0.2, 0.3, 0.4], [1, 2, 3], [1, 1, 2]],
            "y": [[0.1, 0.1, 0.2], [3, 1, 2], [2, 1, 2]],
            "z": [[0.1, 0.1, 0.2], [3, 1, 2], [2, 1, 2]],
        }
    )
    assert function(array, attrs=SOME_ATTRS).attrs is SOME_ATTRS
    assert function(array)._attrs is None
