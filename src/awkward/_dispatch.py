from functools import wraps
from inspect import isgenerator

from awkward._errors import OperationErrorContext
from awkward._typing import Callable, Iterator, TypeAlias, TypeVar

DispatcherType: TypeAlias = Iterator


T = TypeVar("T", bound=Callable)


def high_level_function(func: T) -> T:
    """Decorate a high-level function such that it may be overloaded by third-party array objects"""

    @wraps(func)
    def dispatch(*args, **kwargs):
        # NOTE: this decorator assumes that the operation is exposed under `ak.`
        with OperationErrorContext(f"ak.{func.__qualname__}", args, kwargs):
            gen_or_result = func(*args, **kwargs)
            if isgenerator(gen_or_result):
                try:
                    while True:
                        arg = next(gen_or_result)
                        try:
                            custom_impl = arg.__awkward_function__
                        except AttributeError:
                            continue
                        return custom_impl(dispatch, args, kwargs)
                except StopIteration as err:
                    return err.value

            return func(*args, **kwargs)

    return dispatch
