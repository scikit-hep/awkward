# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.filterwarnings("ignore::awkward.errors.ExperimentalWarning")


@pytest.mark.parametrize("clip", [False, True])
def test_to_dense(clip):
    cpu = ak.Array([[{"a": 1, "b": 2.5}, {"a": None, "b": 3.5}], None, []])
    gpu = ak.to_backend(cpu, "cuda")
    result = ak.to_dense(gpu, clip=clip, fill_value=-1, dtype=np.float32)
    assert type(result) is np.ndarray
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(
        result, ak.to_dense(cpu, clip=clip, fill_value=-1, dtype=np.float32)
    )
