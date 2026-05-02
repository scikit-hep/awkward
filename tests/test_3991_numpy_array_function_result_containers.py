# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

try:
    from numpy._core.overrides import array_function_dispatch
except ImportError:
    from numpy.core.overrides import array_function_dispatch


def test_tuple_of_arrays():
    result = np.histogram(ak.Array([1, 2, 3]), bins=2)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], ak.Array)
    assert result[0].to_list() == [1, 2]
    assert isinstance(result[1], ak.Array)
    assert result[1].to_list() == [1.0, 2.0, 3.0]


def test_list_of_arrays():
    result = np.split(ak.Array([1, 2, 3, 4]), 2)

    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], ak.Array)
    assert result[0].to_list() == [1, 2]
    assert isinstance(result[1], ak.Array)
    assert result[1].to_list() == [3, 4]


def test_namedtuple_of_arrays():
    result = np.linalg.eig(ak.Array([[1.0, 0.0], [0.0, 2.0]]))

    if not hasattr(result, "_fields"):
        pytest.skip("NumPy does not return namedtuple results from np.linalg.eig")

    assert result._fields == ("eigenvalues", "eigenvectors")
    assert isinstance(result.eigenvalues, ak.Array)
    assert result.eigenvalues.to_list() == [1.0, 2.0]
    assert isinstance(result.eigenvectors, ak.Array)
    assert result.eigenvectors.to_list() == [[1.0, 0.0], [0.0, 1.0]]


def test_dict_of_arrays():
    def dict_of_arrays_dispatcher(array):
        return (array,)

    @array_function_dispatch(dict_of_arrays_dispatcher)
    def dict_of_arrays(array):
        array = np.asarray(array)
        return {"array": array, "nested": {"copy": array.copy(), "label": "copy"}}

    result = dict_of_arrays(ak.Array([1, 2, 3]))

    assert type(result) is dict
    assert type(result["nested"]) is dict
    assert isinstance(result["array"], ak.Array)
    assert result["array"].to_list() == [1, 2, 3]
    assert isinstance(result["nested"]["copy"], ak.Array)
    assert result["nested"]["copy"].to_list() == [1, 2, 3]
    assert result["nested"]["label"] == "copy"
