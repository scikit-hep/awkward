# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Callable, Collection, Generator
from functools import wraps
from inspect import isgenerator

from awkward._errors import OperationErrorContext
from awkward._typing import Any, TypeAlias, TypeVar

T = TypeVar("T")
DispatcherType: TypeAlias = "Callable[..., Generator[Collection[Any], None, T]]"
HighLevelType: TypeAlias = "Callable[..., T]"


def high_level_function(
    module: str = "ak", name: str | None = None
) -> Callable[[DispatcherType], HighLevelType]:
    """Decorate a high-level function such that it may be overloaded by third-party array objects"""

    def capture_func(func: DispatcherType) -> HighLevelType:
        if name is None:
            captured_name = func.__qualname__
        else:
            captured_name = name
        return named_high_level_function(func, f"{module}.{captured_name}")

    return capture_func


def named_high_level_function(func: DispatcherType, name: str) -> HighLevelType:
    """Decorate a named high-level function such that it may be overloaded by third-party array objects"""

    @wraps(func)
    def dispatch(*args, **kwargs):
        # NOTE: this decorator assumes that the operation is exposed under `ak.`
        with OperationErrorContext(name, args, kwargs):
            gen_or_result = func(*args, **kwargs)
            if isgenerator(gen_or_result):
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

                        # Future proof the implementation by permitting the `__awkward_function__` to return `NotImplemented`
                        # This may later be used to signal that another overload should be used.
                        if result is NotImplemented:
                            raise NotImplementedError(
                                f"no overload for {name} with arguments {array_likes}"
                            )
                        else:
                            return result

                # Failed to find a custom overload, so resume the original function
                try:
                    next(gen_or_result)
                except StopIteration as err:
                    return err.value
                else:
                    raise AssertionError(
                        "high-level functions should only implement a single yield statement"
                    )

            return gen_or_result

    return dispatch
