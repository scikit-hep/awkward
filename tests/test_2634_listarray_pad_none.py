# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    layout = ak.to_layout(["hi", "bye"])

    def as_32(index):
        return ak.index.Index(index.nplike.astype(index.data, np.int32))

    layout = ak.contents.ListArray(
        as_32(layout.starts),
        as_32(layout.stops),
        layout.content,
        parameters=layout._parameters,
    )
    assert ak.pad_none(layout, 3, axis=0).tolist() == ["hi", "bye", None]
