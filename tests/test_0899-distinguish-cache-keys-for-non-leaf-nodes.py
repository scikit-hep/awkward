# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    form = {
        "class": "RecordArray",
        "contents": {
            "a": {
                "class": "ListOffsetArray64",
                "offsets": "i64",
                "content": {
                    "class": "NumpyArray",
                    "itemsize": 8,
                    "format": "l",
                    "primitive": "int64",
                    "form_key": "node2",
                },
                "form_key": "node0",
            },
            "b": {
                "class": "ListOffsetArray64",
                "offsets": "i64",
                "content": {
                    "class": "NumpyArray",
                    "itemsize": 8,
                    "format": "l",
                    "primitive": "int64",
                    "form_key": "node3",
                },
                "form_key": "node0",
            },
        },
        "form_key": "node1",
    }

    container = {
        "part0-node0-offsets": np.array([0, 2, 3, 3, 6], dtype=np.int64),
        "part0-node2-data": np.array([1, 2, 3, 4, 5, 6], dtype=np.int64),
        "part0-node3-data": np.array([10, 20, 30, 40, 50, 60], dtype=np.int64),
    }

    assert ak.from_buffers(form, 4, container).tolist() == [
        {"a": [1, 2], "b": [10, 20]},
        {"a": [3], "b": [30]},
        {"a": [], "b": []},
        {"a": [4, 5, 6], "b": [40, 50, 60]},
    ]

    assert ak.from_buffers(form, 4, container, lazy=True).tolist() == [
        {"a": [1, 2], "b": [10, 20]},
        {"a": [3], "b": [30]},
        {"a": [], "b": []},
        {"a": [4, 5, 6], "b": [40, 50, 60]},
    ]
