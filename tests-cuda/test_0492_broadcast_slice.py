# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

import awkward as ak

pytest.importorskip("cupy")
pytestmark = pytest.mark.filterwarnings("ignore::awkward.errors.ExperimentalWarning")


@pytest.mark.parametrize(
    ("source", "indices"),
    [
        ([[10, 20], [30]], [[[1, -1, 0]], [[0], []]]),
        ([[{"x": 1}, {"x": None}], [{"x": 2}]], [[[1, 0]], [[0]]]),
        ([[1, 2], None], [[[0, None], None], [[0]]]),
        ([[1], [2]], [[], []]),
    ],
)
def test_cuda(source, indices):
    expected = ak.broadcast_slice(source, indices)
    result = ak.broadcast_slice(
        ak.to_backend(source, "cuda"), ak.to_backend(indices, "cuda")
    )
    assert ak.backend(result) == "cuda"
    assert ak.to_list(result) == ak.to_list(expected)
    assert result.layout.form == expected.layout.form


def test_bounds():
    source = ak.to_backend([[1], [2, 3, 4]], "cuda")
    indices = ak.to_backend([[[1]], [[0]]], "cuda")
    with pytest.raises(IndexError):
        ak.to_list(ak.broadcast_slice(source, indices))
