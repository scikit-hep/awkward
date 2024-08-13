# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import builtins
import sys
import threading
import warnings
from collections.abc import Callable, Collection, Iterable, Mapping
from functools import wraps
from weakref import ref as weak_ref

import numpy

from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._typing import Any, ParamSpec, TypeVar

np = NumpyMetadata.instance()


E = TypeVar("E", bound=Exception)
T = TypeVar("T")
S = TypeVar("S")
P = ParamSpec("P")


class WeakMethodProxy:
    """A proxy for a method of a weakly referenced object"""

    def __init__(self, method):
        self._this = weak_ref(method.__self__)
        self._impl = method.__func__

    def __call__(self, *args, **kwargs):
        this = self._this()
        method = self._impl.__get__(this, type(this))
        return method(*args, **kwargs)


class PartialFunction:
    """Analogue of `functools.partial`, but as a distinct type"""

    __slots__ = ("func", "args", "kwargs")

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.func(*self.args, **self.kwargs)


class KeyError(builtins.KeyError):
    def __str__(self):
        return super(Exception, self).__str__()


class ErrorContext:
    # Any other threads should get a completely independent _slate.
    _slate = threading.local()

    @classmethod
    def primary(cls):
        return cls._slate.__dict__.get("__primary_context__")

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __enter__(self):
        # Make it strictly non-reenterant. Only one ErrorContext (per thread) is primary.
        if self.primary() is None:
            self._slate.__dict__.clear()
            self._slate.__dict__.update(self._kwargs)
            self._slate.__dict__["__primary_context__"] = self

    def __exit__(self, exception_type, exception_value, traceback):
        try:
            # Handle caught exception
            if (
                exception_type is not None
                and issubclass(exception_type, Exception)
                and self.primary() is self
            ):
                self.handle_exception(exception_type, exception_value)
        finally:
            # Step out of the way so that another ErrorContext can become primary.
            if self.primary() is self:
                self._slate.__dict__.clear()

    def handle_exception(self, cls: type[E], exception: E):
        if sys.version_info >= (3, 11, 0, "final"):
            self.decorate_exception(cls, exception)
        else:
            raise self.decorate_exception(cls, exception)

    def decorate_exception(self, cls: type[E], exception: E) -> Exception:
        if sys.version_info >= (3, 11, 0, "final"):
            if issubclass(cls, (NotImplementedError, AssertionError)):
                exception.add_note(
                    "\n\nSee if this has been reported at https://github.com/scikit-hep/awkward/issues"
                )
            else:
                exception.add_note(self.note)
            return exception
        else:
            new_exception: Exception
            if issubclass(cls, (NotImplementedError, AssertionError)):
                # Raise modified exception
                new_exception = cls(
                    str(exception)
                    + "\n\nSee if this has been reported at https://github.com/scikit-hep/awkward/issues"
                )
                new_exception.__cause__ = exception
            elif issubclass(cls, builtins.KeyError):
                new_exception = KeyError(self.format_exception(exception))
                new_exception.__cause__ = exception
            else:
                new_exception = cls(self.format_exception(exception))
                new_exception.__cause__ = exception
            return new_exception

    def format_argument(self, width, value):
        from awkward import contents, highlevel, record

        if isinstance(value, contents.Content):
            return self.format_argument(width, highlevel.Array(value))
        elif isinstance(value, record.Record):
            return self.format_argument(width, highlevel.Record(value))

        valuestr = None
        if isinstance(
            value,
            (
                highlevel.Array,
                highlevel.Record,
                highlevel.ArrayBuilder,
            ),
        ):
            try:
                valuestr = value._repr(width)
            except Exception as err:
                valuestr = f"repr-raised-{type(err).__name__}"

        elif value is None or isinstance(value, (bool, int, float)):
            try:
                valuestr = repr(value)
            except Exception as err:
                valuestr = f"repr-raised-{type(err).__name__}"

        elif isinstance(value, (str, bytes)):
            try:
                if len(value) < 60:
                    valuestr = repr(value)
                else:
                    valuestr = repr(value[:57]) + "..."
            except Exception as err:
                valuestr = f"repr-raised-{type(err).__name__}"

        elif isinstance(value, np.ndarray):
            if not numpy.__version__.startswith("1.13."):  # 'threshold' argument
                prefix = f"{type(value).__module__}.{type(value).__name__}("
                suffix = ")"
                try:
                    valuestr = numpy.array2string(
                        value,
                        max_line_width=width - len(prefix) - len(suffix),
                        threshold=0,
                    ).replace("\n", " ")
                    valuestr = prefix + valuestr + suffix
                except Exception as err:
                    valuestr = f"array2string-raised-{type(err).__name__}"

                if len(valuestr) > width and "..." in valuestr[:-1]:
                    last = valuestr.rfind("...") + 3
                    while last > width:
                        last = valuestr[: last - 3].rfind("...") + 3
                    valuestr = valuestr[:last]

                if len(valuestr) > width:
                    valuestr = valuestr[: width - 3] + "..."

        elif isinstance(value, (Collection, Mapping)) and len(value) < 10000:
            valuestr = repr(value)
            if len(valuestr) > width:
                valuestr = valuestr[: width - 3] + "..."

        if valuestr is None:
            return f"{type(value).__name__}-instance"
        else:
            return valuestr

    def format_exception(self, exception: Exception) -> str:
        raise NotImplementedError

    @property
    def note(self) -> str:
        raise NotImplementedError


class OperationErrorContext(ErrorContext):
    _width = 80 - 8

    def any_backend_is_delayed(
        self, iterable: Iterable, *, depth: int = 1, depth_limit: int = 2
    ) -> bool:
        from awkward._backends.dispatch import backend_of_obj

        for obj in iterable:
            backend = backend_of_obj(obj, default=None)
            # Do we not recognise this as an object with a backend?
            if backend is None:
                # Is this an iterable object, and are we permitted to recurse?
                if isinstance(obj, Collection) and depth != depth_limit:
                    return self.any_backend_is_delayed(
                        obj, depth=depth + 1, depth_limit=depth_limit
                    )
                # Assume not delayed!
                else:
                    return False
            # Eager backends aren't delayed!
            elif backend.nplike.is_eager:
                continue
            else:
                return True
        return False

    def __init__(self, name, args: Iterable[Any], kwargs: Mapping[str, Any]):
        string_args: list[str] | PartialFunction
        string_kwargs: dict[str, str] | PartialFunction
        if self.primary() is None and (
            self.any_backend_is_delayed(args)
            or self.any_backend_is_delayed(kwargs.values())
        ):
            string_args = self._format_args(args)
            string_kwargs = self._format_kwargs(kwargs)
        else:
            # if primary is not None: we won't be setting an ErrorContext
            # if all nplikes are eager: no accumulation of large arrays
            # --> in either case, delay string generation
            string_args = PartialFunction(WeakMethodProxy(self._format_args), args)
            string_kwargs = PartialFunction(
                WeakMethodProxy(self._format_kwargs), kwargs
            )

        super().__init__(
            name=name,
            args=string_args,
            kwargs=string_kwargs,
        )

    def _format_args(self, arguments: Iterable) -> list[str]:
        string_arguments = []
        for value in arguments:
            string_arguments.append(self.format_argument(self._width, value))

        return string_arguments

    def _format_kwargs(self, arguments: Mapping[str, Any]) -> dict[str, str]:
        string_arguments = {}
        for key, value in arguments.items():
            if isinstance(key, str):
                width = self._width - len(key) - 3
            else:
                width = self._width
            string_arguments[key] = self.format_argument(width, value)
        return string_arguments

    @property
    def name(self):
        return self._kwargs["name"]

    @property
    def args(self) -> list:
        out = self._kwargs["args"]
        if isinstance(out, PartialFunction):
            out = self._kwargs["args"] = out()
        return out

    @property
    def kwargs(self) -> dict:
        out = self._kwargs["kwargs"]
        if isinstance(out, PartialFunction):
            out = self._kwargs["kwargs"] = out()
        return out

    def format_exception(self, exception: Exception) -> str:
        return f"{exception}\n{self.note}"

    @property
    def note(self) -> str:
        arguments = []
        for valuestr in self.args:
            arguments.append(f"\n        {valuestr}")
        for name, valuestr in self.kwargs.items():
            if isinstance(name, str):
                arguments.append(f"\n        {name} = {valuestr}")
            else:
                arguments.append(f"\n        {valuestr}")

        extra_line = "" if len(arguments) == 0 else "\n    "
        calling_note = f'{self.name}({"".join(arguments)}{extra_line})'
        return f"""
This error occurred while calling

    {calling_note}"""


class SlicingErrorContext(ErrorContext):
    _width = 80 - 4

    def __init__(self, array, where):
        from awkward._backends.dispatch import backend_of_obj
        from awkward._backends.numpy import NumpyBackend

        numpy_backend = NumpyBackend.instance()
        if self.primary() is not None or all(
            backend_of_obj(x, default=numpy_backend).nplike.is_eager
            for x in (array, where)
        ):
            # if primary is not None: we won't be setting an ErrorContext
            # if all nplikes are eager: no accumulation of large arrays
            # --> in either case, delay string generation
            formatted_array = PartialFunction(
                WeakMethodProxy(self.format_argument), self._width, array
            )
            formatted_slice = PartialFunction(self.format_slice, where)
        else:
            formatted_array = self.format_argument(self._width, array)
            formatted_slice = self.format_slice(where)

        super().__init__(
            array=formatted_array,
            where=formatted_slice,
        )

    @property
    def array(self):
        out = self._kwargs["array"]
        if isinstance(out, PartialFunction):
            out = self._kwargs["array"] = out()
        return out

    @property
    def where(self):
        out = self._kwargs["where"]
        if isinstance(out, PartialFunction):
            out = self._kwargs["where"] = out()
        return out

    def format_exception(self, exception):
        return f"{exception}\n{self.note}"

    @property
    def note(self) -> str:
        return f"""
This error occurred while attempting to slice

    {self.array}

with

    {self.where}"""

    @staticmethod
    def format_slice(x):
        from awkward import contents, highlevel, index, record

        if isinstance(x, slice):
            if x.step is None:
                return "{}:{}".format(
                    "" if x.start is None else x.start,
                    "" if x.stop is None else x.stop,
                )
            else:
                return "{}:{}:{}".format(
                    "" if x.start is None else x.start,
                    "" if x.stop is None else x.stop,
                    x.step,
                )

        elif isinstance(x, tuple):
            return "(" + ", ".join(SlicingErrorContext.format_slice(y) for y in x) + ")"

        elif isinstance(x, index.Index64):
            return str(x.data)

        elif isinstance(x, contents.Content):
            try:
                return str(highlevel.Array(x))
            except Exception:
                return x._repr("    ", "", "")

        elif isinstance(x, record.Record):
            try:
                return str(highlevel.Record(x))
            except Exception:
                return x._repr("    ", "", "")

        else:
            return repr(x)


def index_error(subarray, slicer, details: str | None = None) -> IndexError:
    message = ""
    if details is not None:
        message = f": {details}"

    # Note: returns an error for the caller to raise!
    return IndexError(
        f"cannot slice {type(subarray).__name__} (of length {subarray.length}) with {SlicingErrorContext.format_slice(slicer)}{message}"
    )


###############################################################################

# Enable warnings for the Awkward package
warnings.filterwarnings("default", module="awkward.*")


def deprecate(
    message,
    version,
    date=None,
    will_be="an error",
    category=DeprecationWarning,
    stacklevel=2,
):
    if date is None:
        date = ""
    else:
        date = " (target date: " + date + ")"
    warning = f"""In version {version}{date}, this will be {will_be}.
To raise these warnings as errors (and get stack traces to find out where they're called), run
    import warnings
    warnings.filterwarnings("error", module="awkward.*")
after the first `import awkward` or use `@pytest.mark.filterwarnings("error:::awkward.*")` in pytest.
Issue: {message}."""
    warnings.warn(warning, category, stacklevel=stacklevel + 1)


def with_operation_context(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        # NOTE: this decorator assumes that the operation is exposed under `ak.`
        with OperationErrorContext(f"ak.{func.__qualname__}", args, kwargs):
            return func(*args, **kwargs)

    return wrapper
