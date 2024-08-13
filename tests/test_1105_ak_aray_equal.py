# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def test_array_equal():
    assert ak.array_equal(
        ak.Array([[1, 2], [], [3, 4, 5]]),
        ak.Array([[1, 2], [], [3, 4, 5]]),
    )

    # TODO: More test cases!
