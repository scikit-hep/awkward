from __future__ import annotations

import types
from typing import TypeVar

import awkward as ak
from awkward import _errors, highlevel, nplikes

numpy = nplikes.Numpy()


_is_registered = False


def register_and_check():
    """
    Register Awkward Array node types with Jax's tree mechanism.
    """
    try:
        import jax

    except ModuleNotFoundError:
        raise ak._errors.wrap_error(
            ModuleNotFoundError(
                """install the 'jax' package with:

        python3 -m pip install jax jaxlib

    or

        conda install -c conda-forge jax jaxlib
    """
            )
        ) from None

    _register(jax)


T = TypeVar("T", bound="type[highlevel.Array | highlevel.Record]")


def register_behavior_class(cls: T) -> T:
    """
    Args:
        cls: behavior class to register with Jax

    Return the behavior class, after registering it with Jax.

    """
    jax = assert_registered()
    import awkward._connect.jax as jax_connect

    jax.tree_util.register_pytree_node(
        cls,
        jax_connect.jax_flatten_highlevel,
        jax_connect.jax_unflatten_highlevel,
    )
    return cls


def _register(jax: types.ModuleType):
    """
    Register Awkward Array node types with Jax's tree mechanism.
    """

    # Let's only do this once
    global _is_registered
    if _is_registered:
        return

    try:
        import awkward._connect.jax as jax_connect

        for cls in [
            ak.contents.BitMaskedArray,
            ak.contents.ByteMaskedArray,
            ak.contents.EmptyArray,
            ak.contents.IndexedArray,
            ak.contents.IndexedOptionArray,
            ak.contents.NumpyArray,
            ak.contents.ListArray,
            ak.contents.ListOffsetArray,
            ak.contents.RecordArray,
            ak.contents.UnionArray,
            ak.contents.UnmaskedArray,
            ak.record.Record,
        ]:
            jax.tree_util.register_pytree_node(
                cls,
                cls.jax_flatten,
                cls.jax_unflatten,
            )

        for cls in [ak.highlevel.Array, ak.highlevel.Record]:
            jax.tree_util.register_pytree_node(
                cls,
                jax_connect.jax_flatten_highlevel,
                jax_connect.jax_unflatten_highlevel,
            )
    finally:
        _is_registered = True


def assert_registered():
    """Ensure that Jax integration is registered. Raise a RuntimeError if not."""
    if not _is_registered:
        raise _errors.wrap_error(
            RuntimeError("Jax features require `ak.jax.register()`")
        )


def import_jax():
    """Ensure that Jax integration is registered, and return the Jax module. Raise a RuntimeError if not."""
    assert_registered()
    import jax

    return jax
