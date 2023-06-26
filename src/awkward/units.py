# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import annotations

from threading import RLock

_lock = RLock()
_unit_registry = None
_checked_for_pint = False


def set_unit_registry(registry):
    global _unit_registry
    with _lock:
        _unit_registry = registry


def get_unit_registry():
    with _lock:
        _register_if_available()
        return _unit_registry


def _register_if_available():
    global _checked_for_pint, _unit_registry
    with _lock:
        if _checked_for_pint:
            return

        try:
            import pint
        except ModuleNotFoundError:
            return
        else:
            _unit_registry = pint.UnitRegistry()
        finally:
            _checked_for_pint = True


def register_and_check():
    """
    Build `pint` unit registry
    """
    try:
        import pint  # noqa: F401

    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            """install the 'pint' package with:

        python3 -m pip install pint

    or

        conda install -c conda-forge pint
    """
        ) from None

    _register_if_available()
