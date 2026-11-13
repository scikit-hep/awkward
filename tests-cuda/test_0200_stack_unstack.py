# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

import awkward as ak

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.filterwarnings("ignore::awkward.errors.ExperimentalWarning")


@pytest.mark.parametrize("dropna", [True, False])
def test_stack(dropna):
    array = ak.Array([[(1, 2), (3, None)], [], [(4, 5, 6)]])
    result = ak.stack(ak.to_backend(array, "cuda"), dropna=dropna)
    assert ak.backend(result) == "cuda"
    assert (
        ak.to_backend(result, "cpu").to_list()
        == ak.stack(array, dropna=dropna).to_list()
    )


@pytest.mark.parametrize("size", [0, 2, 4])
def test_unstack(size):
    array = ak.Array([[[1, 2], []], [], [[3]]])
    result = ak.unstack(ak.to_backend(array, "cuda"), size=size)
    assert ak.backend(result) == "cuda"
    assert (
        ak.to_backend(result, "cpu").to_list() == ak.unstack(array, size=size).to_list()
    )


def test_ragged_width_requires_size():
    array = ak.to_backend(ak.Array([[1, 2], [3]]), "cuda")
    with pytest.raises(ValueError, match="explicit size"):
        ak.unstack(array)


def test_regular_width_uses_metadata():
    array = ak.Array(cp.arange(12).reshape(3, 4))
    assert ak.to_backend(ak.unstack(array), "cpu").to_list() == [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9, 10, 11),
    ]
