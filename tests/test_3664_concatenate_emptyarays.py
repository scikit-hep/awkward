# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def test():
    ak.concatenate([ak.Array([None]) for _ in range(5000)], axis=0)
    ak.concatenate([ak.Array([{"x": None}]) for _ in range(5000)], axis=0)
    ak.concatenate([ak.Array([{"x": i, "y": None}]) for i in range(5000)], axis=0)
