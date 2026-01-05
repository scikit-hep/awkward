# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

# taken from test_2719_typetracer_buffer_key.py
form = ak.forms.from_dict(
    {
        "class": "RecordArray",
        "fields": ["x"],
        "contents": [
            {
                "class": "ListOffsetArray",
                "offsets": "i64",
                "content": {
                    "class": "NumpyArray",
                    "primitive": "int64",
                    "inner_shape": [],
                    "parameters": {},
                    "form_key": "x.list.content",
                },
                "parameters": {},
                "form_key": "x.list",
            }
        ],
        "parameters": {},
    }
)


def test_buffer_keys_on_virtual_arrays():
    buffers = {
        "x.list-offsets": lambda: np.array([0, 2, 3, 5]),
        "x.list.content-data": lambda: np.array([1, 2, 3, 4, 5]),
    }

    va = ak.from_buffers(form, 3, buffers, buffer_key="{form_key}-{attribute}")

    assert va.layout.content("x").offsets.data.buffer_key == "x.list-offsets"
    assert va.layout.content("x").content.data.buffer_key == "x.list.content-data"


def test_buffer_keys_on_placeholder_arrays():
    buffers = {
        "x.list-offsets": ak._nplikes.placeholder.PlaceholderArray(
            shape=(4,),
            dtype=np.int64,
            buffer_key="x.list-offsets",
            nplike=ak._nplikes.numpy.Numpy.instance(),
        ),
        "x.list.content-data": ak._nplikes.placeholder.PlaceholderArray(
            shape=(5,),
            dtype=np.int64,
            buffer_key="x.list.content-data",
            nplike=ak._nplikes.numpy.Numpy.instance(),
        ),
    }

    pa = ak.from_buffers(form, 3, buffers, buffer_key="{form_key}-{attribute}")

    assert pa.layout.content("x").offsets.data.buffer_key == "x.list-offsets"
    assert pa.layout.content("x").content.data.buffer_key == "x.list.content-data"

    with pytest.raises(
        RuntimeError,
        match=r"Awkward Array tried to access a buffer at 'x.list-offsets'",
    ):
        pa.layout.content("x").offsets.data.materialize()

    with pytest.raises(
        RuntimeError,
        match=r"Awkward Array tried to access a buffer at 'x.list.content-data'",
    ):
        pa.layout.content("x").content.data.materialize()
