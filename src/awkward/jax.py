from __future__ import annotations

import types

import awkward as ak

_is_registered = False


def register():
    """
    Register Awkward Array node types with Jax's tree mechanism.
    """
    # Let's only do this once
    global _is_registered
    if _is_registered:
        return

    import jax

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
            cls._jax_flatten,
            cls._jax_unflatten,
        )
    _is_registered = True


def import_jax() -> types.ModuleType:
    """
    Import jax and return the module, or raise a helpful error message if it is not available.
    """
    try:
        import jax

    except ModuleNotFoundError:
        raise ak._errors.wrap_error(
            ModuleNotFoundError(
                """install the 'numba' package with:

        python3 -m pip install jax jaxlib

    or

        conda install -c conda-forge jax jaxlib
    """
            )
        ) from None

    register()
    return jax
