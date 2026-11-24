# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

import awkward as ak

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.filterwarnings("ignore::awkward.errors.ExperimentalWarning")


@pytest.mark.parametrize("drop_none", [True, False])
def test_tuples_to_lists(drop_none):
    array = ak.Array([[(1, 2), (3, None)], [], [(4, 5, 6)]])
    result = ak.tuples_to_lists(
        ak.to_backend(array, "cuda"), drop_none=drop_none, pad=not drop_none
    )
    assert ak.backend(result) == "cuda"
    assert (
        ak.to_backend(result, "cpu").to_list()
        == ak.tuples_to_lists(array, drop_none=drop_none, pad=not drop_none).to_list()
    )


@pytest.mark.parametrize("size", [0, 2, 4])
def test_lists_to_tuples(size):
    array = ak.Array([[[1, 2], []], [], [[3]]])
    result = ak.lists_to_tuples(ak.to_backend(array, "cuda"), size=size)
    assert ak.backend(result) == "cuda"
    assert (
        ak.to_backend(result, "cpu").to_list()
        == ak.lists_to_tuples(array, size=size).to_list()
    )


def test_ragged_width_requires_size():
    array = ak.to_backend(ak.Array([[1, 2], [3]]), "cuda")
    with pytest.raises(ValueError, match="explicit size"):
        ak.lists_to_tuples(array)


def test_regular_width_uses_metadata():
    array = ak.Array(cp.arange(12).reshape(3, 4))
    assert ak.to_backend(ak.lists_to_tuples(array), "cpu").to_list() == [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9, 10, 11),
    ]


@pytest.mark.parametrize("size", [0, 1, 3, 6])
def test_regular_field_carries_after_slicing(size):
    source = ak.Array(cp.arange(20).reshape(5, 4))[1:4]
    result = ak.lists_to_tuples(source, size=size)
    expected = ak.lists_to_tuples(ak.to_backend(source, "cpu"), size=size)
    assert ak.to_backend(result, "cpu").to_list() == expected.to_list()
