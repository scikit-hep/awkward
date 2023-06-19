from functools import wraps
from inspect import signature

from awkward._errors import OperationErrorContext
from awkward._typing import Callable, Iterator, TypeAlias, TypeVar

DispatcherType: TypeAlias = Iterator


T = TypeVar("T", bound=Callable)


def high_level_function(dispatcher=None):
    """Decorate a high-level function such that it may be overloaded by third-party array objects"""

    def decorator(func: T) -> T:
        # Let's check the signature!
        if not (dispatcher is None or signature(func) == signature(dispatcher)):
            raise RuntimeError(
                f"Array dispatcher for {func.__qualname__} is incompatible with implementation"
            )

        @wraps(func)
        def dispatch(*args, **kwargs):
            # NOTE: this decorator assumes that the operation is exposed under `ak.`
            with OperationErrorContext(f"ak.{func.__qualname__}", args, kwargs):
                if dispatcher is not None:
                    dispatched = tuple(dispatcher(*args, **kwargs))
                    for arg in dispatched:
                        try:
                            custom_impl = arg.__awkward_function__
                        except AttributeError:
                            continue
                        return custom_impl(dispatch, dispatched, args, kwargs)

                return func(*args, **kwargs)

        return dispatch

    return decorator
