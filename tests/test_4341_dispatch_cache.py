# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._backends.dispatch import (
    _backend_lookup_factories,
    _type_to_backend_lookup,
    backend_of_obj,
    register_backend_lookup_factory,
    regularize_backend,
)
from awkward._nplikes.dispatch import (
    _nplike_classes,
    _type_to_nplike,
    nplike_of_obj,
    register_nplike,
)
from awkward._nplikes.numpy import Numpy


class Unrecognised:
    pass


@pytest.fixture
def pristine_registries():
    """Restore the module-level factory lists and lookup caches afterwards."""
    factories = _backend_lookup_factories[:]
    nplikes = _nplike_classes[:]
    try:
        yield
    finally:
        _backend_lookup_factories[:] = factories
        _nplike_classes[:] = nplikes
        _type_to_backend_lookup.clear()
        _type_to_nplike.clear()


def test_unrecognised_type_raises_every_time(pristine_registries):
    obj = Unrecognised()
    # the negative result is cached, but must keep raising rather than be swallowed
    for _ in range(2):
        with pytest.raises(TypeError, match="cannot find backend for Unrecognised"):
            backend_of_obj(obj)
        with pytest.raises(TypeError, match="cannot find nplike for Unrecognised"):
            nplike_of_obj(obj)
    assert _type_to_backend_lookup[Unrecognised] is None
    assert _type_to_nplike[Unrecognised] is None


@pytest.mark.parametrize("obj", [Unrecognised(), [1, 2, 3], 1, "hi"])
def test_default_is_returned_instead_of_raising(obj):
    sentinel = object()
    assert backend_of_obj(obj, default=sentinel) is sentinel
    assert nplike_of_obj(obj, default=sentinel) is sentinel


def test_recognised_types_still_resolve():
    array = np.arange(3)
    cpu = regularize_backend("cpu")
    assert backend_of_obj(array) is cpu
    assert backend_of_obj(ak.Array([[1.1, 2.2], [3.3]])) is cpu
    assert nplike_of_obj(array) is Numpy.instance()
    # the memoised second call gives back the very same objects
    assert backend_of_obj(array) is backend_of_obj(np.arange(5))
    assert nplike_of_obj(array) is nplike_of_obj(np.arange(5))


def test_register_backend_lookup_factory_invalidates_cache(pristine_registries):
    obj = Unrecognised()
    cpu = regularize_backend("cpu")
    with pytest.raises(TypeError):
        backend_of_obj(obj)

    def factory(cls):
        if issubclass(cls, Unrecognised):
            return lambda obj: cpu

    register_backend_lookup_factory(factory)
    assert Unrecognised not in _type_to_backend_lookup
    assert backend_of_obj(obj) is cpu


def test_register_nplike_invalidates_cache(pristine_registries):
    obj = Unrecognised()
    with pytest.raises(TypeError):
        nplike_of_obj(obj)

    class FakeNplike:
        @classmethod
        def is_own_array_type(cls, type_):
            return issubclass(type_, Unrecognised)

        @classmethod
        def instance(cls):
            return Numpy.instance()

    register_nplike(FakeNplike)
    assert Unrecognised not in _type_to_nplike
    assert nplike_of_obj(obj) is Numpy.instance()
