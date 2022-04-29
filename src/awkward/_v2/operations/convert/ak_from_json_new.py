# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pathlib
from urllib.parse import urlparse

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def from_json(
    source,
    line_delimited=False,
    schema=None,
    nan_string=None,
    infinity_string=None,
    minus_infinity_string=None,
    complex_record_fields=None,
    buffersize=65536,
    initial=1024,
    resize=1.5,
    highlevel=True,
    behavior=None,
):
    """
    Args:
        source (bytes/str, pathlib.Path, or file-like object): Data source of the
            JSON-formatted string(s). If bytes/str, the string is parsed. If a
            `pathlib.Path`, a file with that name is opened, parsed, and closed.
            If that path has a URI protocol (like "https://" or "s3://"), this
            function attempts to open the file with the fsspec library. If a
            file-like object with a `read` method, this function reads from the
            object, but does not close it.
        line_delimited (bool): If False, a single JSON document is read as an
            entire array or record. If True, this function reads line-delimited
            JSON into an array (regardless of how many there are). The line
            delimiter is not actually checked, so it may be `"\n"`, `"\r\n"`
            or anything else.
        schema (None, JSON str or equivalent lists/dicts): If None, the data type
            is discovered while parsing. If a JSONSchema, that schema is used to
            parse the JSON more quickly by skipping type-discovery.
        nan_string (None or str): If not None, strings with this value will be
            interpreted as floating-point NaN values.
        infinity_string (None or str): If not None, strings with this value will
            be interpreted as floating-point positive infinity values.
        minus_infinity_string (None or str): If not None, strings with this value
            will be interpreted as floating-point negative infinity values.
        complex_record_fields (None or (str, str)): If not None, defines a pair of
            field names to interpret 2-field records as complex numbers.
        buffersize (int): Number of bytes in each read from source: larger
            values use more memory but read less frequently. (Python GIL is released
            between read events.)
        initial (int): Initial size (in bytes) of buffers used by
            #ak.layout.ArrayBuilder (see #ak.layout.ArrayBuilderOptions).
        resize (float): Resize multiplier for buffers used by
            #ak.layout.ArrayBuilder (see #ak.layout.ArrayBuilderOptions);
            should be strictly greater than 1.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts a JSON string into an Awkward Array.

    FIXME: needs documentation.

    See also #ak.to_json.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.from_json",
        dict(
            source=source,
            line_delimited=line_delimited,
            schema=schema,
            nan_string=nan_string,
            infinity_string=infinity_string,
            minus_infinity_string=minus_infinity_string,
            complex_record_fields=complex_record_fields,
            buffersize=buffersize,
            initial=initial,
            resize=resize,
            highlevel=highlevel,
            behavior=behavior,
        ),
    ):
        if schema is None:
            return _no_schema(
                source,
                line_delimited,
                nan_string,
                infinity_string,
                minus_infinity_string,
                complex_record_fields,
                buffersize,
                initial,
                resize,
                highlevel,
                behavior,
            )

        else:
            raise ak._v2._util.error(NotImplementedError)


class _BytesReader:
    __slots__ = ("data", "current")

    def __init__(self, data):
        self.data = data
        self.current = 0

    def read(self, num_bytes):
        before = self.current
        self.current += num_bytes
        return self.data[before : self.current]

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        pass


class _NoContextManager:
    def __init__(self, file):
        self.file = file

    def __enter__(self):
        return self.file

    def __exit__(self, exception_type, exception_value, exception_traceback):
        pass


def _get_reader(source):
    if not isinstance(source, pathlib.Path) and isinstance(source, str):
        source = source.encode("utf8", errors="surrogateescape")

    if isinstance(source, bytes):
        return lambda: _BytesReader(source)

    elif isinstance(source, pathlib.Path):
        parsed_url = urlparse(str(source))
        if parsed_url.scheme == "" or parsed_url.netloc == "":
            return lambda: open(source, "rb")  # pylint: disable=R1732
        else:
            import fsspec

            return lambda: fsspec.open(source, "rb").open()

    else:
        return lambda: _NoContextManager(source)


def _record_to_complex(layout, complex_record_fields):
    if complex_record_fields is None:
        return layout

    elif (
        isinstance(complex_record_fields, tuple)
        and len(complex_record_fields) == 2
        and isinstance(complex_record_fields[0], str)
        and isinstance(complex_record_fields[1], str)
    ):

        def action(node, **kwargs):
            if isinstance(node, ak._v2.contents.RecordArray):
                if set(node.fields) == set(complex_record_fields):
                    real = node._getitem_field(complex_record_fields[0])
                    imag = node._getitem_field(complex_record_fields[1])
                    if (
                        isinstance(real, ak._v2.contents.NumpyArray)
                        and len(real.shape) == 1
                        and isinstance(imag, ak._v2.contents.NumpyArray)
                        and len(imag.shape) == 1
                    ):
                        return ak._v2.contents.NumpyArray(
                            node._nplike.asarray(real) + node._nplike.asarray(imag) * 1j
                        )

        return layout.recursively_apply(action)

    else:
        raise ak._v2._util.error(
            TypeError("complex_record_fields must be None or a pair of strings")
        )


def _no_schema(
    source,
    line_delimited,
    nan_string,
    infinity_string,
    minus_infinity_string,
    complex_record_fields,
    buffersize,
    initial,
    resize,
    highlevel,
    behavior,
):
    builder = ak.layout.ArrayBuilder(initial=initial, resize=resize)

    read_one = not line_delimited

    with _get_reader(source)() as obj:
        ak._ext.fromjsonobj(
            obj,
            builder,
            read_one,
            buffersize,
            nan_string,
            infinity_string,
            minus_infinity_string,
        )

    formstr, length, buffers = builder.to_buffers()
    form = ak._v2.forms.from_json(formstr)
    layout = ak._v2.operations.convert.from_buffers(
        form, length, buffers, highlevel=False
    )

    layout = _record_to_complex(layout, complex_record_fields)

    if read_one:
        layout = layout[0]

    if highlevel:
        return ak._v2._util.wrap(layout, behavior, highlevel)
    else:
        return layout
