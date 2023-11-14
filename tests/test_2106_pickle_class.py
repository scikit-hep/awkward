# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import contextlib
import pickle
import sys
import types
import uuid

import pytest

import awkward as ak


@contextlib.contextmanager
def temporary_module():
    name = str(uuid.uuid1()).replace("-", "_")
    module = types.ModuleType(name)
    sys.modules[name] = module
    yield module
    del sys.modules[name]


def impl():
    with temporary_module() as module:
        exec(
            """
import awkward as ak

behavior = {}
@ak.mixin_class(behavior, "my_array")
class MyArray:
    @property
    def meaning_of_life(self):
        return 42
        """,
            module.__dict__,
        )

        array = ak.Array(
            [None, [{"x": [None, 1, "hi"]}]],
            with_name="my_array",
            behavior=module.behavior,
        )
        record = ak.Array(
            [None, [{"x": [None, 1, "hi"]}]],
            with_name="my_array",
            behavior=module.behavior,
        )[1, 0]
        return pickle.dumps(array), pickle.dumps(record)


def global_impl():
    with temporary_module() as module:
        exec(
            """
import awkward as ak

@ak.mixin_class(ak.behavior, "my_array")
class MyArray:
    @property
    def meaning_of_life(self):
        return 42
        """,
            module.__dict__,
        )

        array = ak.Array([None, [{"x": [None, 1, "hi"]}]], with_name="my_array")
        record = ak.Array([None, [{"x": [None, 1, "hi"]}]], with_name="my_array")[1, 0]
        return pickle.dumps(array), pickle.dumps(record)


def test():
    array_data, record_data = impl()

    # Using a custom behavior dictionary will always break unpickling if
    # the objects in the dictionary can't be resolved
    with pytest.raises(ModuleNotFoundError):
        pickle.loads(array_data)

    with pytest.raises(ModuleNotFoundError):
        pickle.loads(record_data)


def test_global(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(ak, "behavior", {})
        array_data, record_data = global_impl()

    # The global ak.behavior is not written to the pickle
    # Awkward can create arrays with missing behavior classes
    other_array = pickle.loads(array_data)
    assert other_array.to_list() == [None, [{"x": [None, 1, "hi"]}]]
    # But it won't have their methods / properties when asked for
    with pytest.raises(AttributeError):
        assert other_array.meaning_of_life == 42

    other_record = pickle.loads(record_data)
    assert other_record.to_list() == {"x": [None, 1, "hi"]}
    # But it won't have their methods / properties when asked for
    with pytest.raises(AttributeError):
        assert other_record.meaning_of_life == 42
