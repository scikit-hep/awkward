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
                array_likes = next(gen_or_result)
                assert isinstance(array_likes, tuple)

                # Permit a third-party array object to intercept the invocation
                for array_like in array_likes:
                    try:
                        custom_impl = array_like.__awkward_function__
                    except AttributeError:
                        continue
                    else:
                        return custom_impl(dispatch, array_likes, args, kwargs)

                # Failed to find a custom overload, so resume the original function
                try:
                    next(gen_or_result)
                except StopIteration as err:
                    return err.value

            return gen_or_result

    return dispatch
