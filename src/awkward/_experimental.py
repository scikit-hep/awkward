# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import os
import warnings
from functools import wraps
from inspect import currentframe

from awkward.errors import ExperimentalWarning

# Trailing separator so that the prefix test cannot match sibling packages
# (awkward_cpp, awkward_pandas, ...).
_PACKAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")


def _find_stack_level() -> int:
    # The stacklevel that attributes a warning to the first frame outside the
    # awkward package. Must be called from the same function that calls
    # warnings.warn: this helper's own frame stands in for that call in the
    # count.
    frame = currentframe()
    n = 0
    try:
        while frame is not None:
            if not frame.f_code.co_filename.startswith(_PACKAGE_DIR):
                return max(n, 1)
            n += 1
            frame = frame.f_back
    finally:
        del frame
    # Every frame is inside awkward, or no frame introspection (non-CPython):
    # attribute to the outermost real frame rather than past the top.
    return max(n - 1, 1)


def experimental(func=None, /):
    """Mark a public function, method, or property accessor as experimental.

    Usable as ``@experimental`` or ``@experimental()``; the parenthesized form
    takes no arguments and is where configuration options can be added later.
    Place the mark directly above the ``def``, below ``@property``,
    ``@classmethod``, ``@staticmethod``, and ``@high_level_function()``, so
    that it always receives a plain function.

    The wrapper issues :class:`awkward.errors.ExperimentalWarning` once per
    process per marked function, on its first call — for a property, on its
    first access — independent of the ``warnings`` filter state.
    """
    if func is None:
        # @experimental() -- the parenthesized form
        return experimental
    if not callable(func):
        raise TypeError(
            "@experimental accepts no arguments; use @experimental or "
            f"@experimental() (got {func!r})"
        )

    message = (
        f"{func.__qualname__} is experimental.\n"
        "    It may change or be removed in any release, without a deprecation period.\n"
        f"    Defined in: {func.__module__}"
    )
    warned = False

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal warned
        if not warned:
            # Set before warning: a filter can turn the warning into a raised
            # error, and at-most-once must hold with the API still usable on
            # later calls.
            warned = True
            warnings.warn(message, ExperimentalWarning, stacklevel=_find_stack_level())
        return func(*args, **kwargs)

    return wrapper
