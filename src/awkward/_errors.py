# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import threading
import warnings
from collections.abc import Callable, Collection, Iterable, Mapping
from functools import wraps

import numpy

from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._typing import Any, ParamSpec, TypeVar
from awkward._util import Sentinel

np = NumpyMetadata.instance()


E = TypeVar("E", bound=Exception)
T = TypeVar("T")
S = TypeVar("S")
P = ParamSpec("P")

# Placeholder for a note fragment whose rendering has been deferred until the
# exception is actually raised.
UNFORMATTED = Sentinel("UNFORMATTED", None)


class ErrorContext:
    # Any other threads should get a completely independent _slate.
    _slate = threading.local()

    @classmethod
    def primary(cls):
        return cls._slate.__dict__.get("__primary_context__")

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def _populate(self) -> None:
        """Build ``self._kwargs``.

        Only the context that actually becomes primary can decorate an
        exception, so subclasses defer their (comparatively expensive)
        argument bookkeeping to here.
        """

    def __enter__(self):
        # Make it strictly non-reenterant. Only one ErrorContext (per thread) is primary.
        slate = self._slate.__dict__
        if slate.get("__primary_context__") is None:
            if self._kwargs is None:
                self._populate()
            slate.clear()
            slate.update(self._kwargs)
            slate["__primary_context__"] = self

    def __exit__(self, exception_type, exception_value, traceback):
        if (
            exception_type is not None
            and issubclass(exception_type, Exception)
            and self.primary() is self
        ):
            # Step out of the way so that another ErrorContext can become primary.
            # Is this necessary to do here? (We're about to raise an exception anyway)
            self._slate.__dict__.clear()
            # Handle caught exception
            raise self.decorate_exception(exception_type, exception_value)
        else:
            # Step out of the way so that another ErrorContext can become primary.
            if self.primary() is self:
                self._slate.__dict__.clear()

    def decorate_exception(self, cls: type[E], exception: E) -> Exception:
        def _add_note(exception: E, note: str) -> E:
            if hasattr(exception, "add_note"):
                exception.add_note(note)
            else:
                exception.__notes__ = [note]
            return exception

        note = self.note
        if issubclass(cls, (NotImplementedError, AssertionError)):
            note = "\n\nSee if this has been reported at https://github.com/scikit-hep/awkward/issues"
        return _add_note(exception, note)

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
                # Only recurse into list/tuple (e.g. the arrays passed to
                # ak.concatenate/ak.zip), never arbitrary containers: a Mapping
                # yields only keys, and a user container (e.g. from_buffers'
                # `container`) may not be iterable or may trigger data reads.
                if isinstance(obj, (list, tuple)) and depth != depth_limit:
                    if self.any_backend_is_delayed(
                        obj, depth=depth + 1, depth_limit=depth_limit
                    ):
                        return True
                # Assume not delayed! Continue checking remaining args.
            # Eager backends aren't delayed!
            elif backend.nplike.is_eager:
                continue
            else:
                return True
        return False

    def __init__(self, name, args: Iterable[Any], kwargs: Mapping[str, Any]):
        self._name = name
        self._raw_args = args
        self._raw_kwargs = kwargs
        self._kwargs = None

    def _populate(self) -> None:
        args = self._raw_args
        kwargs = self._raw_kwargs
        string_args: list[str] | Sentinel
        string_kwargs: dict[str, str] | Sentinel
        if self.any_backend_is_delayed(args) or self.any_backend_is_delayed(
            kwargs.values()
        ):
            string_args = self._format_args(args)
            string_kwargs = self._format_kwargs(kwargs)
        else:
            # if all nplikes are eager: no accumulation of large arrays
            # --> delay string generation
            string_args = string_kwargs = UNFORMATTED

        self._kwargs = {
            "name": self._name,
            "args": string_args,
            "kwargs": string_kwargs,
        }

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
        return self._name

    @property
    def args(self) -> list:
        if self._kwargs is None:
            self._populate()
        out = self._kwargs["args"]
        if out is UNFORMATTED:
            out = self._kwargs["args"] = self._format_args(self._raw_args)
        return out

    @property
    def kwargs(self) -> dict:
        if self._kwargs is None:
            self._populate()
        out = self._kwargs["kwargs"]
        if out is UNFORMATTED:
            out = self._kwargs["kwargs"] = self._format_kwargs(self._raw_kwargs)
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
        calling_note = f"{self.name}({''.join(arguments)}{extra_line})"
        return f"""
This error occurred while calling

    {calling_note}"""


class SlicingErrorContext(ErrorContext):
    _width = 80 - 4

    def __init__(self, array, where):
        self._raw_array = array
        self._raw_where = where
        self._kwargs = None

    def _populate(self) -> None:
        from awkward._backends.dispatch import backend_of_obj

        array = self._raw_array
        where = self._raw_where
        # an object with no backend at all cannot be delayed
        array_backend = backend_of_obj(array, default=None)
        where_backend = backend_of_obj(where, default=None)
        if (array_backend is None or array_backend.nplike.is_eager) and (
            where_backend is None or where_backend.nplike.is_eager
        ):
            # if all nplikes are eager: no accumulation of large arrays
            # --> delay string generation
            formatted_array = formatted_slice = UNFORMATTED
        else:
            formatted_array = self.format_argument(self._width, array)
            formatted_slice = self.format_slice(where)

        self._kwargs = {
            "array": formatted_array,
            "where": formatted_slice,
        }

    @property
    def array(self):
        if self._kwargs is None:
            self._populate()
        out = self._kwargs["array"]
        if out is UNFORMATTED:
            out = self._kwargs["array"] = self.format_argument(
                self._width, self._raw_array
            )
        return out

    @property
    def where(self):
        if self._kwargs is None:
            self._populate()
        out = self._kwargs["where"]
        if out is UNFORMATTED:
            out = self._kwargs["where"] = self.format_slice(self._raw_where)
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
