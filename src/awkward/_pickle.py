"""Interface for plugin-configurable pickle __reduce_ex__ implementation"""
from __future__ import annotations

import sys
import threading
import warnings

from awkward._typing import Any, Protocol, runtime_checkable

if sys.version_info < (3, 12):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


@runtime_checkable
class PickleReducer(Protocol):
    def __call__(self, obj: Any, protocol: int) -> tuple | NotImplemented:
        ...


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


def custom_reduce(obj, protocol) -> tuple | NotImplemented:
    plugin = get_custom_reducer()
    if plugin is None:
        return NotImplemented
    return plugin(obj, protocol)
