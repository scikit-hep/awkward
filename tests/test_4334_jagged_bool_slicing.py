# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak


def test_listoffsetarray():
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6]])
    mask = ak.Array([[True, False, True], [], [False, True], [True]])
    assert array[mask].to_list() == [[1.1, 3.3], [], [5.5], [6.6]]


def test_option_type_content():
    array = ak.Array([[1.1, None, 3.3], [], [None, 5.5]])
    mask = ak.Array([[True, True, False], [], [True, True]])
    assert array[mask].to_list() == [[1.1, None], [], [None, 5.5]]


def test_option_type_mask():
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    mask = ak.Array([[True, None, True], [], [False, True]])
    assert array[mask].to_list() == [[1.1, None, 3.3], [], [5.5]]


def test_all_false():
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    mask = ak.Array([[False, False, False], [], [False, False]])
    assert array[mask].to_list() == [[], [], []]


def test_all_true():
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    mask = ak.Array([[True, True, True], [], [True, True]])
    assert array[mask].to_list() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]


def test_empty_lists():
    array = ak.Array([[], [1.1, 2.2], []])
    mask = ak.Array([[], [False, True], []])
    assert array[mask].to_list() == [[], [2.2], []]

    empty = ak.Array([[]])
    assert empty[ak.Array([[]])].to_list() == [[]]


def test_typetracer():
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    mask = ak.Array([[True, False, True], [], [False, True]])

    concrete = array[mask]
    tracer = array.layout.to_typetracer(forget_length=True)[
        mask.layout.to_typetracer(forget_length=True)
    ]
    assert tracer.form == concrete.layout.form


def test_jax():
    pytest.importorskip("jax")
    ak.jax.register_and_check()

    array = ak.to_backend(ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]), "jax")
    mask = ak.to_backend(ak.Array([[True, False, True], [], [False, True]]), "jax")

    result = array[mask]
    assert ak.backend(result) == "jax"
    assert result.to_list() == [[1.1, 3.3], [], [5.5]]
