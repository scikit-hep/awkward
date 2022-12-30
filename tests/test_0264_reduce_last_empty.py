# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test():
    assert to_list(
        ak.operations.prod(
            ak.highlevel.Array([[[2, 3, 5]], [[7], [11]], [[]]]), axis=-1
        )
    ) == [
        [30],
        [7, 11],
        [1],
    ]

    assert to_list(
        ak.operations.prod(ak.highlevel.Array([[[2, 3, 5]], [[7], [11]], []]), axis=-1)
    ) == [
        [30],
        [7, 11],
        [],
    ]
