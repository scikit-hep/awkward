# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Interface for plugin-configurable pickle __reduce_ex__ implementation"""

from __future__ import annotations

import sys
import threading
import warnings
from collections.abc import Mapping
from contextlib import contextmanager

from awkward._typing import TYPE_CHECKING, Any, JSONMapping, Protocol, runtime_checkable

if sys.version_info < (3, 12):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


if TYPE_CHECKING:
    from awkward._nplikes.shape import ShapeItem
    from awkward.highlevel import Array, Record


@runtime_checkable
class PickleReducer(Protocol):
    def __call__(self, obj: Any, protocol: int) -> tuple | NotImplemented: ...


_register_lock = threading.Lock()
_plugin: PickleReducer | None = None
_is_registered = False


def _load_reduce_plugin():
    best_plugin = None

    for entry_point in importlib_metadata.entry_points(group="awkward.pickle.reduce"):
        plugin = entry_point.load()

        try:
            assert isinstance(plugin, PickleReducer)
        except AssertionError:
            warnings.warn(
                f"Couldn't load `awkward.pickle.reduce` plugin: {entry_point}",
                stacklevel=2,
            )
            continue

        if best_plugin is not None:
            raise RuntimeError(
                "Encountered multiple Awkward pickle reducers under the `awkward.pickle.reduce` entrypoint"
            )
        best_plugin = plugin

    return best_plugin


def get_custom_reducer() -> PickleReducer | None:
    """
    Returns the implementation of a custom __reduce_ex__ function for Awkward
    highlevel objects, or None if none provided
    """
    global _is_registered, _plugin

    with _register_lock:
        if not _is_registered:
            _plugin = _load_reduce_plugin()
            _is_registered = True

        if _plugin is None:
            return None
        else:
            return _plugin


_DISABLE_CUSTOM_REDUCER = False


@contextmanager
def use_builtin_reducer():
    global _DISABLE_CUSTOM_REDUCER
    old_value, _DISABLE_CUSTOM_REDUCER = _DISABLE_CUSTOM_REDUCER, True
    try:
        yield
    finally:
        _DISABLE_CUSTOM_REDUCER = old_value


def custom_reduce(obj, protocol: int) -> tuple | NotImplemented:
    if (plugin := get_custom_reducer()) is None or _DISABLE_CUSTOM_REDUCER:
        return NotImplemented
    else:
        return plugin(obj, protocol)


def unpickle_array_schema_1(
    form_dict: dict,
    length: ShapeItem,
    container: Mapping[str, Any],
    behavior: JSONMapping | None,
    attrs: JSONMapping | None,
) -> Array:
    from awkward.operations.ak_from_buffers import _impl

    return _impl(
        form_dict,
        length,
        container,
        backend="cpu",
        behavior=behavior,
        attrs=attrs,
        highlevel=True,
        buffer_key="{form_key}-{attribute}",
        byteorder="<",
        simplify=False,
    )


def unpickle_record_schema_1(
    form_dict: dict,
    length: ShapeItem,
    container: Mapping[str, Any],
    behavior: JSONMapping | None,
    attrs: JSONMapping | None,
    at: int,
) -> Record:
    from awkward.highlevel import Record
    from awkward.operations.ak_from_buffers import _impl
    from awkward.record import Record as LowLevelRecord

    array_layout = _impl(
        form_dict,
        length,
        container,
        backend="cpu",
        behavior=behavior,
        attrs=attrs,
        highlevel=False,
        buffer_key="{form_key}-{attribute}",
        byteorder="<",
        simplify=False,
    )
    layout = LowLevelRecord(array_layout, at)
    return Record(layout, behavior=behavior, attrs=attrs)
