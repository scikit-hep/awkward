# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pathlib
import pickle

import numpy as np

SAMPLES_DIR = pathlib.Path(__file__).parent / "samples"


def test_numpyarray():
    with open(SAMPLES_DIR / "awkward1-numpyarray.pkl", "rb") as file:
        array = pickle.load(file)
        assert array.to_list() == [1, 2, 3]
        assert pickle.loads(pickle.dumps(array)).layout.form == array.layout.form


def test_partitioned_numpyarray():
    with open(SAMPLES_DIR / "awkward1-partitioned-numpyarray.pkl", "rb") as file:
        array = pickle.load(file)
        assert array.to_list() == [1, 2, 3, 4, 5, 6]
        assert pickle.loads(pickle.dumps(array)).layout.form == array.layout.form


def test_listoffsetarray():
    with open(SAMPLES_DIR / "awkward1-listoffsetarray.pkl", "rb") as file:
        array = pickle.load(file)
        assert array.to_list() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
        assert pickle.loads(pickle.dumps(array)).layout.form == array.layout.form


def test_regulararray():
    with open(SAMPLES_DIR / "awkward1-regulararray.pkl", "rb") as file:
        array = pickle.load(file)
        assert array.to_list() == np.arange(2 * 3 * 5).reshape(2, 3, 5).tolist()
        assert pickle.loads(pickle.dumps(array)).layout.form == array.layout.form


def test_strings():
    with open(SAMPLES_DIR / "awkward1-strings.pkl", "rb") as file:
        array = pickle.load(file)
        assert array.to_list() == ["one", "two", "three"]
        assert pickle.loads(pickle.dumps(array)).layout.form == array.layout.form


def test_emptyarray():
    with open(SAMPLES_DIR / "awkward1-emptyarray.pkl", "rb") as file:
        array = pickle.load(file)
        assert array.to_list() == []
        assert pickle.loads(pickle.dumps(array)).layout.form == array.layout.form


def test_indexedoptionarray():
    with open(SAMPLES_DIR / "awkward1-indexedoptionarray.pkl", "rb") as file:
        array = pickle.load(file)
        assert array.to_list() == [1, 2, None, None, 3]
        assert pickle.loads(pickle.dumps(array)).layout.form == array.layout.form


def test_bytemaskedarray():
    with open(SAMPLES_DIR / "awkward1-bytemaskedarray.pkl", "rb") as file:
        array = pickle.load(file)
        assert array.to_list() == [1, 2, None, None, 5]
        assert pickle.loads(pickle.dumps(array)).layout.form == array.layout.form


def test_unmaskedarray():
    with open(SAMPLES_DIR / "awkward1-unmaskedarray.pkl", "rb") as file:
        array = pickle.load(file)
        assert array.to_list() == [1, 2, 3]
        assert str(array.type) == "3 * ?int64"
        assert pickle.loads(pickle.dumps(array)).layout.form == array.layout.form


def test_recordarray():
    with open(SAMPLES_DIR / "awkward1-recordarray.pkl", "rb") as file:
        array = pickle.load(file)
        assert array.to_list() == [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}]
        assert pickle.loads(pickle.dumps(array)).layout.form == array.layout.form


def test_recordarray_tuple():
    with open(SAMPLES_DIR / "awkward1-recordarray-tuple.pkl", "rb") as file:
        array = pickle.load(file)
        assert array.to_list() == [(1, 1.1), (2, 2.2)]
        assert pickle.loads(pickle.dumps(array)).layout.form == array.layout.form


def test_unionarray():
    with open(SAMPLES_DIR / "awkward1-unionarray.pkl", "rb") as file:
        array = pickle.load(file)
        assert array.to_list() == [1, 2, [3, 4, 5]]
        assert pickle.loads(pickle.dumps(array)).layout.form == array.layout.form
