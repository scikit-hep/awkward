# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pickle

import pytest  # noqa: F401

import awkward as ak


def impl():
    behavior = {}

    @ak.mixin_class(behavior, "my_array")
    class MyArray:
        @property
        def meaning_of_life(self):
            return 42

    array = ak.Array(
        [None, [{"x": [None, 1, "hi"]}]], with_name="my_array", behavior=behavior
    )
    return pickle.dumps(array)


def global_impl():
    @ak.mixin_class(ak.behavior, "my_array")
    class MyArray:
        @property
        def meaning_of_life(self):
            return 42

    array = ak.Array([None, [{"x": [None, 1, "hi"]}]], with_name="my_array")
    return pickle.dumps(array)


def test():
    other = pickle.loads(impl())
    assert other.meaning_of_life == 42


def test_global(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(ak, "behavior", {})
        data = global_impl()

    other = pickle.loads(data)
    assert other.meaning_of_life == 42
