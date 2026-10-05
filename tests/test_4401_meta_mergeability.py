# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import itertools

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.shape import unknown_length
from awkward._nplikes.virtual import VirtualNDArray


def forbidden(*args, **kwargs):
    pytest.fail("metadata comparison must not access buffers or resolve virtual shapes")


@pytest.mark.parametrize(
    "shape",
    [
        (10,),
        (unknown_length,),
        (10, 2),
        (unknown_length, 2),
        (unknown_length, unknown_length),
        (10, 2, 3),
        (unknown_length, 2, unknown_length),
        (0, 0, 3),
    ],
)
@pytest.mark.parametrize("policy", ["same_kind", "equiv", "family"])
def test_virtual_numpy_metadata_only(shape, policy, monkeypatch):
    buffers = [
        VirtualNDArray(
            Numpy.instance(),
            shape=shape,
            dtype=np.dtype(dtype),
            generator=forbidden,
            shape_generator=forbidden,
        )
        for dtype in ("int64", "float64")
    ]
    layouts = [ak.contents.NumpyArray(buffer) for buffer in buffers]
    forms = [
        ak.forms.NumpyForm(dtype, inner_shape=(2,) * (len(shape) - 1))
        for dtype in ("int64", "float64")
    ]
    regular_forms = [form.to_RegularForm() for form in forms]
    # Avoid a wholesale Content -> Form conversion, which can itself resolve
    # unknown inner shapes. The predicate only needs rank and dtype.
    monkeypatch.setattr(ak.contents.Content, "form", property(forbidden))
    monkeypatch.setattr(ak.contents.NumpyArray, "data", property(forbidden))
    monkeypatch.setattr(ak.forms.Form, "length_zero_array", forbidden)
    monkeypatch.setattr(ak.forms.Form, "length_one_array", forbidden)
    monkeypatch.setattr(ak, "from_buffers", forbidden)

    for one, two in itertools.product(
        (layouts[0], forms[0], regular_forms[0]),
        (layouts[1], forms[1], regular_forms[1]),
    ):
        for left, right in ((one, two), (two, one)):
            assert ak._do.mergeable(left, right, mergecastable=policy) == (
                policy == "same_kind"
            )
            assert left._mergeable_next(right, True, policy) == (policy == "same_kind")
    assert all(not buffer.is_materialized for buffer in buffers)


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([1, None], [2.5], True),
        ([[1], []], [[2.5, 3.5]], True),
        ([[1]], [2], False),
        ([{"x": [1], "y": 2}], [{"y": 3.5, "x": [4.5]}], True),
        ([{"x": 1}], [{"y": 2}], False),
        ([(1, 2)], [(3,)], False),
        ([1, "two"], [{"x": 3}], True),
        (["one"], [[1]], False),
        ([], [1], True),
    ],
)
def test_nested_virtual_buffers(left, right, expected, monkeypatch):
    layouts = []
    forms = []
    for data in (left, right):
        form, length, container = ak.to_buffers(ak.Array(data))
        forms.append(form)
        layouts.append(
            ak.from_buffers(
                form, length, dict.fromkeys(container, forbidden), highlevel=False
            )
        )
    monkeypatch.setattr(ak.contents.Content, "form", property(forbidden))
    monkeypatch.setattr(ak.forms.Form, "length_zero_array", forbidden)
    monkeypatch.setattr(ak, "from_buffers", forbidden)
    for one, two in itertools.product((layouts[0], forms[0]), (layouts[1], forms[1])):
        assert ak._do.mergeable(one, two) is expected
        assert ak._do.mergeable(two, one) is expected


@pytest.mark.parametrize("mergebool", [True, False])
@pytest.mark.parametrize("policy", ["same_kind", "equiv", "family"])
def test_virtual_boolean_casting(mergebool, policy):
    layouts = [
        ak.contents.NumpyArray(
            VirtualNDArray(
                Numpy.instance(),
                shape=(unknown_length, unknown_length),
                dtype=np.dtype(dtype),
                generator=forbidden,
                shape_generator=forbidden,
            )
        )
        for dtype in ("bool", "float64")
    ]
    assert ak._do.mergeable(*layouts, mergebool, policy) is mergebool


def test_merging_still_materializes_and_preserves_values():
    calls = []

    def generate():
        calls.append(True)
        return np.arange(6, dtype=np.int64).reshape(2, 3)

    left = ak.contents.NumpyArray(
        VirtualNDArray(
            Numpy.instance(), shape=(2, 3), dtype=np.dtype("int64"), generator=generate
        )
    )
    right = ak.contents.NumpyArray(np.arange(6, 12, dtype=np.int64).reshape(2, 3))
    assert ak._do.mergeable(left, right)
    assert not calls
    result = ak.concatenate([left, right])
    assert result.to_list() == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    assert calls
