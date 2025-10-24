# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
from __future__ import annotations

import numpy as np

import awkward as ak


def test():
    """Check that to_list() does not break when the array is built from buffers
    with a length of type np.int64.
    """
    awk = ak.Array(np.ones((7, 0)))
    form, length, container = ak.to_buffers(ak.to_packed(awk))
    awk_from_buf = ak.from_buffers(form, np.int64(length), container)
    lst = awk_from_buf.to_list()

    assert len(lst) == length
