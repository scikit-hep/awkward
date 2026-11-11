# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import reprlib
import sys
import threading
import warnings
from collections.abc import Callable, Collection, Iterable, Mapping
from functools import cached_property, wraps
from itertools import islice

import numpy

from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._typing import Any, ParamSpec, TypeVar

np = NumpyMetadata.instance()


E = TypeVar("E", bound=Exception)
T = TypeVar("T")
S = TypeVar("S")
P = ParamSpec("P")


class ErrorContext:
    # Any other threads should get a completely independent _slate.
    _slate = threading.local()
    _frozen = False

    @classmethod
    def primary(cls):
        return cls._slate.__dict__.get("__primary_context__")

    @classmethod
    def frozen_primary(cls):
        """The primary context, frozen so that it can outlive the call."""
        context = cls.primary()
        if context is not None and not context._frozen:
            context.freeze()
            context._frozen = True
        return context

    def freeze(self) -> None:
        """Format the note now and drop the references to the raw arguments."""

    def __enter__(self):
        # Make it strictly non-reenterant. Only one ErrorContext (per thread) is primary.
        slate = self._slate.__dict__
        if slate.get("__primary_context__") is None:
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
            try:
                valuestr = _ArgumentRepr(width).repr(value)
            except Exception as err:
                valuestr = f"repr-raised-{type(err).__name__}"
            if len(valuestr) > width:
                valuestr = valuestr[: width - 3] + "..."

        if valuestr is None:
            return f"{type(value).__name__}-instance"
        else:
            return valuestr

    @property
    def note(self) -> str:
        raise NotImplementedError


class _ArgumentRepr(reprlib.Repr):
    """
    Limit container depth and total visits while formatting an argument.
    Ordinary acyclic values that fit the budget retain their repr() text.
    """

    def __init__(self, width):
        super().__init__()
        self.maxlevel = self.maxlist = self.maxtuple = self.maxdict = width
        self.maxset = self.maxfrozenset = self.maxdeque = self.maxarray = width
        self.maxstring = self.maxother = self.maxlong = sys.maxsize
        self._active = set()
        self._remaining = max(width, 1) * 4

    def repr1(self, x, level):
        # A depth bound alone still permits exponential work on branching
        # cycles or shared subtrees. Share one visit budget across the whole
        # argument, and stop cycles along the current traversal path.
        identity = id(x)
        if identity in self._active:
            # Level 0 gives repr()'s own placeholder for a cycle, e.g. "[...]".
            return self._dispatch(x, 0)
        if self._remaining <= 0:
            return "..."
        self._remaining -= 1
        self._active.add(identity)
        try:
            return self._dispatch(x, level)
        finally:
            self._active.remove(identity)

    def _dispatch(self, x, level):
        # reprlib dispatches by type name, which would send a subclass to the
        # unbounded builtin repr(); route any subclass whose repr() is the builtin's.
        for base in (list, tuple, dict, set, frozenset):
            if isinstance(x, base) and type(x).__repr__ is base.__repr__:
                return getattr(self, "repr_" + base.__name__)(x, level)
        return super().repr1(x, level)

    def repr_dict(self, x, level):
        # reprlib sorts keys; repr() keeps insertion order.
        if not x:
            return "{}"
        if level <= 0:
            return "{...}"
        pieces = [
            f"{self.repr1(k, level - 1)}: {self.repr1(v, level - 1)}"
            for k, v in islice(x.items(), self.maxdict)
        ]
        if len(x) > self.maxdict:
            pieces.append("...")
        return "{" + ", ".join(pieces) + "}"

    def repr_set(self, x, level):
        # reprlib sorts members; repr() keeps iteration order.
        name = type(x).__name__
        if not x:
            return f"{name}()"
        if type(x) is set:
            return self._repr_iterable(x, level, "{", "}", self.maxset)
        return self._repr_iterable(x, level, f"{name}({{", "})", self.maxset)

    repr_frozenset = repr_set


class OperationErrorContext(ErrorContext):
    _width = 80 - 8

    def __init__(self, name, args: Iterable[Any], kwargs: Mapping[str, Any]):
        self._name = name
        self._raw_args = args
        self._raw_kwargs = kwargs

    def freeze(self) -> None:
        _ = self.note
        self._raw_args, self._raw_kwargs = (), {}

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

    @cached_property
    def args(self) -> list[str]:
        return self._format_args(self._raw_args)

    @cached_property
    def kwargs(self) -> dict[str, str]:
        return self._format_kwargs(self._raw_kwargs)

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

    def freeze(self) -> None:
        _ = self.note
        self._raw_array = self._raw_where = None

    @cached_property
    def array(self) -> str:
        return self.format_argument(self._width, self._raw_array)

    @cached_property
    def where(self) -> str:
        return self.format_slice(self._raw_where)

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
