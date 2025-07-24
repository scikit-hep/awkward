# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
#
from __future__ import annotations

import pytest

import awkward as ak


def test():
    ak.concatenate([ak.to_regular([[i, i], [i, i]]) for i in range(64)], axis=1)
    ak.concatenate([ak.to_regular([[i, i], [i, i]]) for i in range(127)], axis=1)
    ak.concatenate([ak.to_regular([[i, i], [i, i]]) for i in range(128)], axis=1)
    ak.concatenate([ak.to_regular([[i, i], [i, i]]) for i in range(129)], axis=1)
    ak.concatenate([ak.to_regular([[i, i], [i, i]]) for i in range(256)], axis=1)

    ak.concatenate(
        [ak.to_regular([[i, str(i)], [str(i), i]]) for i in range(64)], axis=1
    )
    ak.concatenate(
        [ak.to_regular([[i, str(i)], [str(i), i]]) for i in range(127)], axis=1
    )
    ak.concatenate(
        [ak.to_regular([[i, str(i)], [str(i), i]]) for i in range(128)], axis=1
    )
    ak.concatenate(
        [ak.to_regular([[i, str(i)], [str(i), i]]) for i in range(129)], axis=1
    )

    ak.concatenate(
        [ak.to_regular([[i, str(i)], [str(i), i]]) for i in range(64)], axis=0
    )
    with pytest.raises(
        ValueError, match="UnionArray cannot have more than 128 content types"
    ):
        ak.concatenate(
            [ak.to_regular([[i, str(i)], [str(i), i]]) for i in range(65)], axis=0
        )
