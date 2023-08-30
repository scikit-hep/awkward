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


@runtime_checkable
class PickleReducerPlugin(PickleReducer, Protocol):
    rank: int


_register_lock = threading.Lock()
_plugins: tuple[PickleReducerPlugin, ...] | None = None
_is_registered = False


def _load_reduce_plugins() -> tuple[PickleReducerPlugin, ...]:
    plugins: list[PickleReducerPlugin] = []

    for entry_point in importlib_metadata.entry_points(group="awkward.pickle.reduce"):
        plugin = entry_point.load()

        try:
            assert isinstance(plugin, PickleReducerPlugin)
        except AssertionError:
            warnings.warn(
                f"Couldn't load `awkward.pickle.reduce` plugin: {entry_point}",
                stacklevel=2,
            )
            continue

        plugins.append(plugin)

    plugins.sort(key=lambda x: x.rank, reverse=True)
    return tuple(plugins)


def get_custom_reducers() -> tuple[PickleReducer, ...] | None:
    """
    Returns the implementation of a custom __reduce_ex__ function for Awkward
    highlevel objects, or None if none provided
    """
    global _is_registered, _plugins

    with _register_lock:
        if not _is_registered:
            _plugins = _load_reduce_plugins()
            _is_registered = True

    return _plugins


def custom_reduce(obj, protocol: int) -> tuple | NotImplemented:
    for plugin in get_custom_reducers():
        result = plugin(obj, protocol)
        if result is not NotImplemented:
            return result
    return NotImplemented
