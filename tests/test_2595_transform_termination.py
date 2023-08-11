# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest

import awkward as ak


def test_single_no_termination():
    def transform(layout, **kwargs):
        pass

    with pytest.raises(RuntimeError, match=r"expected to terminate"):
        ak.transform(transform, [{"x": 1}], expect_return_value=True)

    result = ak.transform(transform, [{"x": 1}], expect_return_value=False)
    assert result.to_list() == [{"x": 1}]

    result = ak.transform(
        transform, [{"x": 1}], expect_return_value=True, return_value="none"
    )
    assert result is None


def test_single_termination():
    def transform(layout, **kwargs):
        if layout.is_numpy:
            return ak.contents.NumpyArray(layout.data * 2)

    result = ak.transform(transform, [{"x": 1}], expect_return_value=True)
    assert result.to_list() == [{"x": 2}]

    result = ak.transform(transform, [{"x": 1}], expect_return_value=False)
    assert result.to_list() == [{"x": 2}]

    ak.transform(transform, [{"x": 1}], expect_return_value=True, return_value="none")
    assert result.to_list() == [{"x": 2}]


def test_many_no_termination():
    def transform(layout, **kwargs):
        pass

    with pytest.raises(RuntimeError, match=r"expected to terminate"):
        ak.transform(transform, [{"x": 1}], [2], expect_return_value=True)

    result = ak.transform(transform, [{"x": 1}], [2], expect_return_value=False)
    assert result[0].to_list() == [{"x": 1}]
    assert result[1].to_list() == [{"x": 2}]

    result = ak.transform(
        transform, [{"x": 1}], [2], expect_return_value=True, return_value="none"
    )
    assert result is None


def test_many_termination():
    def transform(inputs, **kwargs):
        if all(layout.is_numpy for layout in inputs):
            return tuple([ak.contents.NumpyArray(layout.data * 2) for layout in inputs])

    result = ak.transform(transform, [{"x": 1}], [2], expect_return_value=True)
    assert result[0].to_list() == [{"x": 2}]
    assert result[1].to_list() == [{"x": 4}]

    result = ak.transform(transform, [{"x": 1}], [2], expect_return_value=False)
    assert result[0].to_list() == [{"x": 2}]
    assert result[1].to_list() == [{"x": 4}]

    ak.transform(
        transform, [{"x": 1}], [2], expect_return_value=True, return_value="none"
    )
    assert result[0].to_list() == [{"x": 2}]
    assert result[1].to_list() == [{"x": 4}]
