# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import os
import warnings
from collections.abc import Callable
from functools import wraps
from inspect import currentframe

from awkward._typing import ParamSpec, TypeVar, overload
from awkward.errors import ExperimentalWarning

P = ParamSpec("P")
T = TypeVar("T")


@overload
def experimental(func: Callable[P, T], /) -> Callable[P, T]: ...


@overload
def experimental(
    func: None = None, /
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


def experimental(
    func: Callable[P, T] | None = None, /
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """Mark a public function, method, or property accessor as experimental.

    An experimental API is public and usable, but exempt from the stability
    guarantee: it may change or be removed in any release, without a
    deprecation period. The mark makes that status visible at runtime, so that
    users learn it from the code they run, not only from the documentation.
    The policy behind the mark is described in
    https://github.com/scikit-hep/awkward/issues/4197.

    Use it as ``@experimental`` or ``@experimental()``; the parenthesized form
    takes no arguments and is reserved for future configuration options.
    Place the mark directly above the ``def``, below other decorators such as
    ``@property``, ``@classmethod``, ``@staticmethod``, and
    ``@high_level_function()``, so that it always receives a plain function.

    The wrapper issues :class:`awkward.errors.ExperimentalWarning` once per
    process per marked function, on the first call — for a property, on the
    first access. This once-only behavior does not depend on the ``warnings``
    filter state: even a filter that turns the warning into an error leaves
    the marked API usable from the second call on.

    Args:
        func (callable or None): The function to mark, in the
            ``@experimental`` form; None, in the ``@experimental()`` form.

    Returns:
        The function wrapped to warn on first use, with the signature, name,
        and docstring of ``func`` preserved; in the parenthesized form, the
        decorator itself.

    Examples:
        Mark a function:

        >>> @experimental
        ... def fuse(*arrays): ...

        The first call issues the warning; later calls are silent:

        >>> fuse()
        <stdin>:1: ExperimentalWarning: fuse is experimental.
            It may change or be removed in any release, without a deprecation period.
            Defined in: __main__

        Mark a property accessor, below ``@property`` so that the mark
        receives the plain getter; the warning is issued on first access:

        >>> class Plan:
        ...     @property
        ...     @experimental
        ...     def cache(self):
        ...         return self._cache

        Users who accept the instability can silence the warning:

        >>> import warnings
        >>> warnings.filterwarnings("ignore", category=ak.errors.ExperimentalWarning)
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
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        nonlocal warned
        if not warned:
            # Set before warning: a filter can turn the warning into a raised
            # error, and at-most-once must hold with the API still usable on
            # later calls.
            warned = True
            warnings.warn(message, ExperimentalWarning, stacklevel=_find_stack_level())
        return func(*args, **kwargs)

    return wrapper


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
