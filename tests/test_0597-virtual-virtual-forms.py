# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    array = ak.Array(
        [[{"a": 1, "b": [1, 2, 3]}], [{"a": 1, "b": [4, 5]}, {"a": 4, "b": [2]}]]
    )
    array_new = ak.Array(
        ak.layout.ListOffsetArray64(
            array.layout.offsets,
            ak.layout.RecordArray(
                {
                    "a": array.layout.content["a"],
                    "b": ak.layout.ListArray64(
                        array.layout.content["b"].offsets[:-1],
                        array.layout.content["b"].offsets[1:],
                        array.layout.content["b"].content,
                    ),
                }
            ),
        )
    )
    form, length, container = ak.to_buffers(array_new)
    reconstituted = ak.from_buffers(form, length, container, lazy=True)
    assert reconstituted.tolist() == array_new.tolist()
