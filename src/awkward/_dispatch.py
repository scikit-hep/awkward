# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import contextlib
from collections.abc import Callable, Collection, Generator, Mapping
from functools import wraps
from inspect import isgeneratorfunction

from awkward._errors import with_named_operation_context
from awkward._requirements import (
    dependency_specification_context,
    with_has_requirements,
)
from awkward._typing import Any, TypeAlias, TypeVar

# First, we parse the dependency information into an immutable specification
T = TypeVar("T")
DispatcherType: TypeAlias = "Callable[..., Generator[Collection[Any], None, T]]"
DispatchedType: TypeAlias = "Callable[..., T]"


def with_type_dispatch(
    func: DispatcherType,
    internal_dispatch_context_factory=contextlib.nullcontext,
) -> DispatchedType:
    if isgeneratorfunction(func):

        @wraps(func)
        def dispatch(*args, **kwargs):
            gen_or_result = func(*args, **kwargs)
            array_likes = next(gen_or_result)
            assert isinstance(array_likes, Collection)

            # Permit a third-party array object to intercept the invocation
            for array_like in array_likes:
                try:
                    custom_impl = array_like.__awkward_function__
                except AttributeError:
                    continue
                else:
                    result = custom_impl(dispatch, array_likes, args, kwargs)

                    # Future-proof the implementation by permitting the `__awkward_function__`
                    # to return `NotImplemented`. This may later be used to signal that another
                    # overload should be used.
                    if result is NotImplemented:
                        raise NotImplementedError
                    else:
                        return result

            # Failed to find a custom overload, so resume the original function
            with internal_dispatch_context_factory():
                try:
                    next(gen_or_result)
                except StopIteration as err:
                    return err.value
                else:
                    raise AssertionError(
                        "high-level functions should only implement a single yield statement"
                    )
    else:

        @wraps(func)
        def dispatch(*args, **kwargs):
            with internal_dispatch_context_factory():
                return func(*args, **kwargs)

    return dispatch


def high_level_function(
    module: str = "ak",
    name: str | None = None,
    *,
    dependencies: Collection[str | Mapping[str, Collection[str]]] | None = None,
) -> Callable[[DispatcherType], DispatchedType]:
    """Decorate a high-level function such that it may be overloaded by third-party array objects"""

    def decorator(func: DispatcherType) -> DispatchedType:
        if name is None:
            captured_name = func.__qualname__
        else:
            captured_name = name

        # Context manager for dispatches that use internal implementation
        def context_factory():
            spec = func_has_requirements.get_specification()
            return dependency_specification_context(spec)

        func_has_requirements = with_has_requirements(
            with_named_operation_context(
                with_type_dispatch(func, context_factory),
                f"{module}.{captured_name}",
            )
        )

        return func_has_requirements

    return decorator
