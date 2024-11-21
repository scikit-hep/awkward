from __future__ import annotations

import awkward as ak
import numpy as np


def test():
    ak.concatenate([ak.Array([i])[:, np.newaxis] for i in range(127)], axis=1)

    ak.concatenate([ak.Array([i])[:, np.newaxis] for i in range(128)], axis=1)