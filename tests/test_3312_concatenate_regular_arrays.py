from __future__ import annotations

import numpy as np

import awkward as ak


def test():
    ak.concatenate([ak.Array([i])[:, np.newaxis] for i in range(127)], axis=1)

    ak.concatenate([ak.Array([i])[:, np.newaxis] for i in range(128)], axis=1)
