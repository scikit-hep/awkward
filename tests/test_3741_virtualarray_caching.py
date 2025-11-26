# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.shape import unknown_length


@pytest.mark.parametrize("offsets_length", [5, unknown_length])
@pytest.mark.parametrize("content_length", [9, unknown_length])
def test(offsets_length, content_length):
    offset_generator = lambda: np.array([0, 2, 4, 5, 6], dtype=np.int64)  # noqa: E731
    data_generator = lambda: np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)  # noqa: E731
    buffers = {"node0-offsets": offset_generator, "node1-data": data_generator}
    form = ak.forms.ListOffsetForm(
        "i64", ak.forms.NumpyForm("int64", form_key="node1"), form_key="node0"
    )

    array = ak.from_buffers(form, 4, buffers, enable_virtualarray_caching=True)
    assert array.to_list() == [[1, 2], [3, 4], [5], [6]]
    assert array.layout.is_all_materialized
    assert ak.materialize(array).to_list() == [[1, 2], [3, 4], [5], [6]]

    array = ak.from_buffers(form, 4, buffers, enable_virtualarray_caching=False)
    assert array.to_list() == [[1, 2], [3, 4], [5], [6]]
    assert not array.layout.is_any_materialized
    assert ak.materialize(array).to_list() == [[1, 2], [3, 4], [5], [6]]

    array = ak.from_buffers(
        form, 4, buffers, enable_virtualarray_caching=lambda form_key, attribute: True
    )
    assert array.to_list() == [[1, 2], [3, 4], [5], [6]]
    assert array.layout.is_all_materialized
    assert ak.materialize(array).to_list() == [[1, 2], [3, 4], [5], [6]]

    array = ak.from_buffers(
        form, 4, buffers, enable_virtualarray_caching=lambda form_key, attribute: False
    )
    assert array.to_list() == [[1, 2], [3, 4], [5], [6]]
    assert not array.layout.is_any_materialized
    assert ak.materialize(array).to_list() == [[1, 2], [3, 4], [5], [6]]

    array = ak.from_buffers(
        form,
        4,
        buffers,
        enable_virtualarray_caching=lambda form_key, attribute: attribute != "data",
    )
    assert array.to_list() == [[1, 2], [3, 4], [5], [6]]
    assert array.layout.offsets.is_all_materialized
    assert not array.layout.content.is_any_materialized
    assert ak.materialize(array).to_list() == [[1, 2], [3, 4], [5], [6]]

    array = ak.from_buffers(
        form,
        4,
        buffers,
        enable_virtualarray_caching=lambda form_key, attribute: attribute == "data",
    )
    assert array.to_list() == [[1, 2], [3, 4], [5], [6]]
    assert not array.layout.offsets.is_any_materialized
    assert array.layout.content.is_all_materialized
    assert ak.materialize(array).to_list() == [[1, 2], [3, 4], [5], [6]]

    array = ak.from_buffers(
        form,
        4,
        buffers,
        enable_virtualarray_caching=lambda form_key, attribute: attribute != "offsets",
    )
    assert array.to_list() == [[1, 2], [3, 4], [5], [6]]
    assert not array.layout.offsets.is_any_materialized
    assert array.layout.content.is_all_materialized
    assert ak.materialize(array).to_list() == [[1, 2], [3, 4], [5], [6]]

    array = ak.from_buffers(
        form,
        4,
        buffers,
        enable_virtualarray_caching=lambda form_key, attribute: attribute == "offsets",
    )
    assert array.to_list() == [[1, 2], [3, 4], [5], [6]]
    assert array.layout.offsets.is_all_materialized
    assert ak.materialize(array).to_list() == [[1, 2], [3, 4], [5], [6]]
