# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import itertools

import numpy as np
import pytest

import awkward as ak

pytest.importorskip("cupy")


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_union_simplification_matches_cpu(order):
    values = [[1], ["a"], [2, "b"]]
    source = ak.contents.UnionArray(
        ak.index.Index8(np.array([0, 1, 2], dtype=np.int8)),
        ak.index.Index64(np.array([0, 0, 0], dtype=np.int64)),
        [ak.to_layout([values[i]]) for i in order],
    )
    gpu = ak.to_backend(source, "cuda", highlevel=False)
    result = ak.contents.UnionArray.simplified(gpu.tags, gpu.index, gpu.contents)
    assert not result.is_union
    assert ak.to_backend(result, "cpu").to_list() == [values[i] for i in order]
