from __future__ import annotations

import enum
import threading
import weakref

import awkward as ak
from awkward import _errors, _nplikes, highlevel
from awkward.typing import TypeVar

numpy = _nplikes.Numpy.instance()


def assert_never(arg) -> None:
    raise ak._errors.wrap_error(AssertionError(f"this should never be run: {arg}"))


class _RegistrationState(enum.Enum):
    INIT = enum.auto()
    SUCCESS = enum.auto()
    FAILED = enum.auto()


_registration_lock = threading.RLock()
_registration_state = _RegistrationState.INIT


def register_and_check():
    """
    Register Awkward Array node types with JAX's tree mechanism.
    """
    try:
        import jax  # noqa: F401

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

    _register()


HighLevelType = TypeVar(
    "HighLevelType", bound="type[highlevel.Array | highlevel.Record]"
)


_known_highlevel_classes = weakref.WeakSet([highlevel.Array, highlevel.Record])


def register_behavior_class(cls: HighLevelType):
    """
    Args:
        cls: behavior class to register with JAX

    Register the behavior class with JAX, if JAX integration is enabled. Otherwise,
    queue the type for subsequent registration when/if JAX is registered.
    """
    # Acquire lock so that we know registration has completed
    with _registration_lock:
        if _registration_state == _RegistrationState.SUCCESS:
            # Safe to invoke JAX code here
            import awkward._connect.jax as jax_connect

            jax_connect.register_pytree_class(cls)
        else:
            _known_highlevel_classes.add(cls)


def _register():
    """
    Register Awkward Array node types with JAX's tree mechanism.
    """
    global _registration_state
    # Require that no threads are trying to register before checking this flag
    with _registration_lock:
        if _registration_state != _RegistrationState.INIT:
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
                jax_connect.register_pytree_class(cls)

            for cls in _known_highlevel_classes:
                jax_connect.register_pytree_class(cls)
        except Exception:
            _registration_state = _RegistrationState.FAILED
            raise  # noqa: AK101
        else:
            _registration_state = _RegistrationState.SUCCESS


def assert_registered():
    """Ensure that JAX integration is registered. Raise a RuntimeError if not."""
    with _registration_lock:
        if _registration_state == _RegistrationState.INIT:
            raise _errors.wrap_error(
                RuntimeError("JAX features require `ak.jax.register_and_check()`")
            )
        elif _registration_state == _RegistrationState.FAILED:
            raise _errors.wrap_error(
                RuntimeError(
                    "JAX features require `ak.jax.register_and_check()`, "
                    "but the last call to `ak.jax.register_and_check()` did not succeed. "
                    "Please look for a traceback to identify the error."
                )
            )
        elif _registration_state == _RegistrationState.SUCCESS:
            return

        assert_never(_registration_state)


def import_jax():
    """Ensure that JAX integration is registered, and return the JAX module. Raise a RuntimeError if not."""
    assert_registered()
    import jax

    return jax
