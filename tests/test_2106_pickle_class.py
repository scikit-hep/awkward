# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
import contextlib
import pickle
import sys
import types
import uuid

import pytest  # noqa: F401

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
        return pickle.dumps(array)


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
        return pickle.dumps(array)


def test():
    data = impl()
    with pytest.raises(ModuleNotFoundError):
        pickle.loads(data)


def test_global(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(ak, "behavior", {})
        data = global_impl()

    other = pickle.loads(data)
    with pytest.raises(AttributeError):
        assert other.meaning_of_life == 42
