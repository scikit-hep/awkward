# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import threading
import traceback
import warnings
from collections.abc import Mapping, Sequence

from awkward import _util, nplikes

np = nplikes.NumpyMetadata.instance()


class ErrorContext:
    # Any other threads should get a completely independent _slate.
    _slate = threading.local()

    _width = 80

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
        # Step out of the way so that another ErrorContext can become primary.
        if self.primary() is self:
            self._slate.__dict__.clear()

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
            import numpy

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

        elif isinstance(value, (Sequence, Mapping)) and len(value) < 10000:
            valuestr = repr(value)
            if len(valuestr) > width:
                valuestr = valuestr[: width - 3] + "..."

        if valuestr is None:
            return f"{type(value).__name__}-instance"
        else:
            return valuestr

    def format_exception(self, exception: Exception) -> str:
        raise NotImplementedError


class OperationErrorContext(ErrorContext):
    def __init__(self, name, arguments):
        string_arguments = {}
        for key, value in arguments.items():
            if _util.isstr(key):
                width = self._width - 8 - len(key) - 3
            else:
                width = self._width - 8

            string_arguments[key] = self.format_argument(width, value)

        super().__init__(
            name=name,
            arguments=string_arguments,
            traceback=traceback.extract_stack(limit=3)[0],
        )

    @property
    def name(self):
        return self._kwargs["name"]

    @property
    def arguments(self):
        return self._kwargs["arguments"]

    @property
    def traceback(self):
        return self._kwargs["traceback"]

    def format_exception(self, exception):
        tb = self.traceback
        try:
            location = f" (from {tb.filename}, line {tb.lineno})"
        except Exception:
            location = ""

        arguments = []
        for name, valuestr in self.arguments.items():
            if _util.isstr(name):
                arguments.append(f"\n        {name} = {valuestr}")
            else:
                arguments.append(f"\n        {valuestr}")

        extra_line = "" if len(arguments) == 0 else "\n    "
        return f"""while calling{location}

    {self.name}({"".join(arguments)}{extra_line})

Error details: {str(exception)}"""


class SlicingErrorContext(ErrorContext):
    def __init__(self, array, where):
        super().__init__(
            array=self.format_argument(self._width - 4, array),
            where=self.format_slice(where),
            traceback=traceback.extract_stack(limit=3)[0],
        )

    @property
    def array(self):
        return self._kwargs["array"]

    @property
    def where(self):
        return self._kwargs["where"]

    @property
    def traceback(self):
        return self._kwargs["traceback"]

    def format_exception(self, exception):
        tb = self.traceback
        try:
            location = f" (from {tb.filename}, line {tb.lineno})"
        except Exception:
            location = ""

        if _util.isstr(exception):
            message = exception
        else:
            message = f"Error details: {str(exception)}"

        return f"""while attempting to slice{location}

    {self.array}

with

    {self.where}

{message}"""

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


def wrap_error(
    exception: Exception | type[Exception], error_context: ErrorContext = None
) -> Exception:
    """
    Args:
        exception: exception object, or exception type to instantiate
        error_context: context in which error was raised

    Wrap the given exception, instantiating it if needed, to ensure meaningful error messages.
    """
    if isinstance(exception, type) and issubclass(exception, Exception):
        try:
            exception = exception()
        except Exception:
            return exception

    if isinstance(exception, (NotImplementedError, AssertionError)):
        return type(exception)(
            str(exception)
            + "\n\nSee if this has been reported at https://github.com/scikit-hep/awkward-1.0/issues"
        )

    if error_context is None:
        error_context = ErrorContext.primary()

    if isinstance(error_context, ErrorContext):
        # Note: returns an error for the caller to raise!
        return type(exception)(error_context.format_exception(exception))
    else:
        # Note: returns an error for the caller to raise!
        return exception


def index_error(subarray, slicer, details: str = None) -> IndexError:
    detailsstr = ""
    if details is not None:
        detailsstr = f"""

Error details: {details}."""

    error_context = ErrorContext.primary()
    if not isinstance(error_context, SlicingErrorContext):
        # Note: returns an error for the caller to raise!
        return IndexError(
            f"cannot slice {type(subarray).__name__} with {SlicingErrorContext.format_slice(slicer)}{detailsstr}"
        )

    else:
        # Note: returns an error for the caller to raise!
        return IndexError(
            error_context.format_exception(
                f"at inner {type(subarray).__name__} of length {subarray.length}, using sub-slice {error_context.format_slice(slicer)}.{detailsstr}"
            )
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
    warning = """In version {}{}, this will be {}.
To raise these warnings as errors (and get stack traces to find out where they're called), run
    import warnings
    warnings.filterwarnings("error", module="awkward.*")
after the first `import awkward` or use `@pytest.mark.filterwarnings("error:::awkward.*")` in pytest.
Issue: {}.""".format(
        version, date, will_be, message
    )
    warnings.warn(warning, category, stacklevel=stacklevel + 1)
