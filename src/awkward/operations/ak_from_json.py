# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import json
import pathlib
from collections.abc import Iterable, Sized
from urllib.parse import urlparse

from awkward_cpp.lib import _ext

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()
numpy = ak._nplikes.Numpy.instance()


def from_json(
    source,
    *,
    line_delimited=False,
    schema=None,
    nan_string=None,
    posinf_string=None,
    neginf_string=None,
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
            If that path has a URI protocol (like `"https://"` or `"s3://"`), this
            function attempts to open the file with the fsspec library. If a
            file-like object with a `read` method, this function reads from the
            object, but does not close it.
        line_delimited (bool): If False, a single JSON document is read as an
            entire array or record. If True, this function reads line-delimited
            JSON into an array (regardless of how many there are). The line
            delimiter is not actually checked, so it may be `"\\n"`, `"\\r\\n"`
            or anything else.
        schema (None, JSON str or equivalent lists/dicts): If None, the data type
            is discovered while parsing. If a JSONSchema
            ([json-schema.org](https://json-schema.org/)), that schema is used to
            parse the JSON more quickly by skipping type-discovery.
        nan_string (None or str): If not None, strings with this value will be
            interpreted as floating-point NaN values.
        posinf_string (None or str): If not None, strings with this value will
            be interpreted as floating-point positive infinity values.
        neginf_string (None or str): If not None, strings with this value
            will be interpreted as floating-point negative infinity values.
        complex_record_fields (None or (str, str)): If not None, defines a pair of
            field names to interpret 2-field records as complex numbers.
        buffersize (int): Number of bytes in each read from source: larger
            values use more memory but read less frequently. (Python GIL is
            released before and after read events.)
        initial (int): Initial size (in bytes) of buffers used by the `ak::ArrayBuilder`.
        resize (float): Resize multiplier for buffers used by the `ak::ArrayBuilder`;
            should be strictly greater than 1.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts a JSON string into an Awkward Array.

    There are a few different dichotomies in JSON-reading; all of the combinations
    are supported:

    * Reading from in-memory str/bytes, on-disk or over-network file, or an
      arbitrary Python object with a `read(num_bytes)` method.
    * Reading a single JSON document or a sequence of line-delimited documents.
    * Unknown schema (slow and general) or with a provided JSONSchema (fast, but
      not all possible cases are supported).
    * Conversion of strings representing not-a-number, plus and minus infinity
      into the appropriate floating-point numbers.
    * Conversion of records with a real and imaginary part into complex numbers.

    Non-JSON features not allowed, including literals for not-a-number or infinite
    numbers; they must be quoted strings for `nan_string`, `posinf_string`, and
    `neginf_string` to recognize. The document or line-delimited documents must
    adhere to the strict [JSON schema](https://www.json.org/).

    Sources
    =======

    In-memory strings or bytes are simply passed as the first argument:

        >>> ak.from_json("[[1.1, 2.2, 3.3], [], [4.4, 5.5]]")
        <Array [[1.1, 2.2, 3.3], [], [4.4, 5.5]] type='3 * var * float64'>

    File names/paths need to be wrapped in `pathlib.Path`, and remote files are
    recognized by URI protocol (like `"https://"` or `"s3://"`) and handled by fsspec
    (which must be installed).

        >>> import pathlib
        >>> with open("tmp.json", "w") as file:
        ...     file.write("[[1.1, 2.2, 3.3], [], [4.4, 5.5]]")
        ...
        33
        >>> ak.from_json(pathlib.Path("tmp.json"))
        <Array [[1.1, 2.2, 3.3], [], [4.4, 5.5]] type='3 * var * float64'>

    And any object with a `read(num_bytes)` method can be used as the `source`.

        >>> class HasReadMethod:
        ...     def __init__(self, data):
        ...         self.bytes = data.encode()
        ...         self.pos = 0
        ...     def read(self, num_bytes):
        ...         start = self.pos
        ...         self.pos += num_bytes
        ...         return self.bytes[start:self.pos]
        ...
        >>> filelike_obj = HasReadMethod("[[1.1, 2.2, 3.3], [], [4.4, 5.5]]")
        >>> ak.from_json(filelike_obj)
        <Array [[1.1, 2.2, 3.3], [], [4.4, 5.5]] type='3 * var * float64'>

    If this function opens a file or network connection (because it is passed as
    a `pathlib.Path`), then this function will also close that file or connection.

    If this function is provided a file-like object with a `read(num_bytes)` method,
    this function will not close it. (It might not even have a `close` method.)

    Data structures
    ===============

    This function interprets JSON arrays and JSON objects in the same way that
    #ak.from_iter interprets Python lists and Python dicts. It could be used as a
    synonym for Python's `json.loads` followed by #ak.from_iter, but the direct
    JSON-reading is faster (especially with a schema) and uses less memory.

    Consider

        >>> import json
        >>> json_data = "[[1.1, 2.2, 3.3], [], [4.4, 5.5]]"
        >>> ak.from_iter(json.loads(json_data))
        <Array [[1.1, 2.2, 3.3], [], [4.4, 5.5]] type='3 * var * float64'>
        >>> ak.from_json(json_data)
        <Array [[1.1, 2.2, 3.3], [], [4.4, 5.5]] type='3 * var * float64'>

    and

        >>> json_data = '{"x": 1.1, "y": [1, 2]}'
        >>> ak.from_iter(json.loads(json_data))
        <Record {x: 1.1, y: [1, 2]} type='{x: float64, y: var * int64}'>
        >>> ak.from_json(json_data)
        <Record {x: 1.1, y: [1, 2]} type='{x: float64, y: var * int64}'>

    As shown above, reading JSON may result in #ak.Array or #ak.Record, but line-delimited
    (`line_delimited=True`) only results in #ak.Array:

        >>> ak.from_json(
        ...     '{"x": 1.1, "y": [1]}\\n{"x": 2.2, "y": [1, 2]}\\n{"x": 3.3, "y": [1, 2, 3]}',
        ...     line_delimited=True,
        ... )
        <Array [{x: 1.1, y: [1]}, ..., {x: 3.3, ...}] type='3 * {x: float64, y: var...'>

    Even arrays of length zero:

        >>> ak.from_json("", line_delimited=True)
        <Array [] type='0 * unknown'>

    Note that JSON interpreted with `line_delimited` doesn't actually need delimiters
    between JSON documents or an absence of delimiters within each document. Parsing
    with `line_delimited=True` continues to the end of a JSON document and starts
    again with the next JSON document. It may be necessary to require actual delimiters
    between and never within JSON documents to split a large source for
    parallel-processing, but that consideration is beyond this function.

    If a JSONSchema is provided, the schema describes the structure of the JSON
    document, regardless of whether there's only one of them (may be an #ak.Record)
    or many of them (must be an #ak.Array).

        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "x": {"type": "number"},
        ...         "y": {"type": "array", "items": {"type": "integer"}},
        ...     },
        ...     "required": ["x", "y"],
        ... }

        >>> ak.from_json(
        ...     '{"x": 1.1, "y": [1, 2, 3]}',
        ...     schema=schema,
        ... )
        <Record {x: 1.1, y: [1, ..., 3]} type='{x: float64, y: var * int64}'>

        >>> ak.from_json(
        ...     '{"x": 1.1, "y": [1]}\\n{"x": 2.2, "y": [1, 2]}\\n{"x": 3.3, "y": [1, 2, 3]}',
        ...     schema=schema,
        ...     line_delimited=True,
        ... )
        <Array [{x: 1.1, y: [1]}, ..., {x: 3.3, ...}] type='3 * {x: float64, y: var...'>

    All numbers in the final array are signed 64-bit (integers and floating-point).

    JSONSchemas
    ===========

    This function supports a subset of JSONSchema (see the
    [JSONSchema specification](https://json-schema.org/)). The schemas may be passed
    as JSON text or as Python lists and dicts representing JSON, but the following
    conditions apply:

    * The root of the schema must be `"type": "array"` or `"type": "object"`.
    * Every level must have a `"type"`, which can only name one type (as a string
      or length-1 list) or one type and `"null"` (as a length-2 list).
    * `"type": "boolean"` \u2192 1-byte boolean values.
    * `"type": "integer"` \u2192 8-byte integer values. If a part of the schema
      is declared to have integer type but the JSON numbers are expressed as
      floating-point, such as `3.14`, `3.0`, or `3e0`, this function raises an
      error.
    * `"type": "number"` \u2192 8-byte floating-point values. If used with
      this function's `nan_string`, `posinf_string`, and/or `neginf_string`, the
      value in the JSON could be a string, as long as it matches one of these
      three.
    * `"type": "string"` \u2192 UTF-8 encoded strings. All JSON escape sequences are
      supported. Remember that the `source` data are ASCII; Unicode is derived from
      "`\\uXXXX`" escape sequences. If an `"enum"` is given, strings are represented
      as categorical values (#ak.contents.IndexedArray or #ak.contents.IndexedOptionArray).
    * `"type": "array"` \u2192 nested lists. The `"items"` must be specified. If
      `"minItems"` and `"maxItems"` are specified and equal to each other, the
      list has regular-type (#ak.types.RegularType); otherwise, it has variable-length
      type (#ak.types.ListType).
    * `"type": "object"` \u2192 nested records. The `"properties"` must be specified,
      and any properties in the data not described by `"properties"` will not
      appear in the output.

    Substitutions for non-finite and complex numbers
    ================================================

    JSON doesn't support not-a-number values, infinite values, or complex number
    types (as in numbers with a real and imaginary part). Some work-arounds use
    non-JSON syntax, but this function converts valid JSON into these numbers with
    user-specified rules.

    The `nan_string`, `posinf_string`, and `neginf_string` convert quoted strings
    into floating-point numbers. You can specify what these strings are.

        >>> ak.from_json(
        ...     '[1, 2, "nan", "inf", "-inf"]',
        ...     nan_string="nan",
        ...     posinf_string="inf",
        ...     neginf_string="-inf",
        ... )
        <Array [1, 2, nan, inf, -inf] type='5 * float64'>

    Without these rules, the array would be interpreted as a union of numbers and
    strings:

        >>> ak.from_json(
        ...     '[1, 2, "nan", "inf", "-inf"]',
        ... )
        <Array [1, 2, 'nan', 'inf', '-inf'] type='5 * union[int64, string]'>

    When combined with a JSONSchema, you need to say that these values have type
    `"number"`, not a union of strings and numbers (i.e. the conversion is performed
    *before* schema-validation). Note that they can't be `"integer"`, since
    not-a-number and infinite values are only possible for floating-point numbers.

        >>> ak.from_json(
        ...     '[1, 2, "nan", "inf", "-inf"]',
        ...     nan_string="nan",
        ...     posinf_string="inf",
        ...     neginf_string="-inf",
        ...     schema={"type": "array", "items": {"type": "number"}}
        ... )
        <Array [1, 2, nan, inf, -inf] type='5 * float64'>

    The `complex_record_fields` is a 2-tuple of field names (strings) of objects
    to identify as the real and imaginary parts of complex numbers. Complex number
    representations in JSON vary, though most are JSON objects with real and
    imaginary parts and possibly other fields. Any other fields will be excluded
    from the output array.

        >>> ak.from_json(
        ...     '[{"r": 1, "i": 1.1, "other": ""}, {"r": 2, "i": 2.2, "other": ""}]',
        ...     complex_record_fields=("r", "i"),
        ... )
        <Array [1+1.1j, 2+2.2j] type='2 * complex128'>

    Without this rule, the array would be interpreted as an array of records:

        >>> ak.from_json(
        ...     '[{"r": 1, "i": 1.1, "other": ""}, {"r": 2, "i": 2.2, "other": ""}]',
        ... )
        <Array [{r: 1, i: 1.1, other: ''}, {...}] type='2 * {r: int64, i: float64, ...'>

    When combined with a JSONSchema, you need to specify the object type (i.e. the
    conversion is performed *after* schema-validation). Note that even the fields
    that will be ignored by `complex_record_fields` need to be specified.

        >>> ak.from_json(
        ...     '[{"r": 1, "i": 1.1, "other": ""}, {"r": 2, "i": 2.2, "other": ""}]',
        ...     complex_record_fields=("r", "i"),
        ...     schema={
        ...         "type": "array",
        ...         "items": {
        ...             "type": "object",
        ...             "properties": {
        ...                 "r": {"type": "number"},
        ...                 "i": {"type": "number"},
        ...                 "other": {"type": "string"},
        ...             },
        ...             "required": ["r", "i"],
        ...         },
        ...     },
        ... )
        <Array [1+1.1j, 2+2.2j] type='2 * complex128'>

    See also #ak.to_json.
    """
    with ak._errors.OperationErrorContext(
        "ak.from_json",
        dict(
            source=source,
            line_delimited=line_delimited,
            schema=schema,
            nan_string=nan_string,
            posinf_string=posinf_string,
            neginf_string=neginf_string,
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
                posinf_string,
                neginf_string,
                complex_record_fields,
                buffersize,
                initial,
                resize,
                highlevel,
                behavior,
            )

        else:
            return _yes_schema(
                source,
                line_delimited,
                schema,
                nan_string,
                posinf_string,
                neginf_string,
                complex_record_fields,
                buffersize,
                initial,
                resize,
                highlevel,
                behavior,
            )


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
        isinstance(complex_record_fields, Sized)
        and isinstance(complex_record_fields, Iterable)
        and len(complex_record_fields) == 2
        and isinstance(complex_record_fields[0], str)
        and isinstance(complex_record_fields[1], str)
    ):

        def action(node, **kwargs):
            if isinstance(node, ak.contents.RecordArray):
                if set(complex_record_fields).issubset(node.fields):
                    real = node._getitem_field(complex_record_fields[0])
                    imag = node._getitem_field(complex_record_fields[1])
                    if (
                        isinstance(real, ak.contents.NumpyArray)
                        and len(real.shape) == 1
                        and issubclass(real.dtype.type, (np.integer, np.floating))
                        and isinstance(imag, ak.contents.NumpyArray)
                        and len(imag.shape) == 1
                        and issubclass(imag.dtype.type, (np.integer, np.floating))
                    ):
                        with numpy._module.errstate(invalid="ignore"):
                            return ak.contents.NumpyArray(
                                node.backend.nplike.asarray(real)
                                + node.backend.nplike.asarray(imag) * 1j
                            )
                    else:
                        raise ak._errors.wrap_error(
                            ValueError(
                                f"expected record with fields {complex_record_fields[0]!r} and {complex_record_fields[1]!r} to have integer or floating point types, not {str(real.form.type)!r} and {str(imag.form.type)!r}"
                            )
                        )

        return ak._do.recursively_apply(layout, action)

    else:
        raise ak._errors.wrap_error(
            TypeError("complex_record_fields must be None or a pair of strings")
        )


def _no_schema(
    source,
    line_delimited,
    nan_string,
    posinf_string,
    neginf_string,
    complex_record_fields,
    buffersize,
    initial,
    resize,
    highlevel,
    behavior,
):
    builder = _ext.ArrayBuilder(initial=initial, resize=resize)

    read_one = not line_delimited

    with _get_reader(source)() as obj:
        try:
            _ext.fromjsonobj(
                obj,
                builder,
                read_one,
                buffersize,
                nan_string,
                posinf_string,
                neginf_string,
            )
        except Exception as err:
            raise ak._errors.wrap_error(ValueError(str(err))) from None

    formstr, length, buffers = builder.to_buffers()
    form = ak.forms.from_json(formstr)
    layout = ak.operations.from_buffers(form, length, buffers, highlevel=False)

    layout = _record_to_complex(layout, complex_record_fields)

    if read_one:
        layout = layout[0]

    if highlevel and isinstance(layout, (ak.contents.Content, ak.record.Record)):
        return ak._util.wrap(layout, behavior, highlevel)
    else:
        return layout


def _yes_schema(
    source,
    line_delimited,
    schema,
    nan_string,
    posinf_string,
    neginf_string,
    complex_record_fields,
    buffersize,
    initial,
    resize,
    highlevel,
    behavior,
):
    if isinstance(schema, (bytes, str)):
        schema = json.loads(schema)

    if not isinstance(schema, dict):
        raise ak._errors.wrap_error(
            TypeError(f"unrecognized JSONSchema: expected dict, got {schema!r}")
        )

    container = {}
    instructions = []

    if schema.get("type") == "array":
        if "items" not in schema:
            raise ak._errors.wrap_error(
                TypeError("JSONSchema type is not concrete: array without items")
            )

        instructions.append(["TopLevelArray"])
        form = build_assembly(schema["items"], container, instructions)
        is_record = False

    elif schema.get("type") == "object":
        form = build_assembly(schema, container, instructions)
        is_record = True

    else:
        raise ak._errors.wrap_error(
            TypeError(
                "only 'array' and 'object' types supported at the JSONSchema root"
            )
        )

    read_one = not line_delimited

    with _get_reader(source)() as obj:
        try:
            length = _ext.fromjsonobj_schema(
                obj,
                container,
                read_one,
                buffersize,
                nan_string,
                posinf_string,
                neginf_string,
                json.dumps(instructions),
                initial,
                resize,
            )
        except Exception as err:
            raise ak._errors.wrap_error(ValueError(str(err))) from None

    layout = ak.operations.from_buffers(form, length, container, highlevel=False)
    layout = _record_to_complex(layout, complex_record_fields)

    if is_record and read_one:
        layout = layout[0]

    if highlevel and isinstance(layout, (ak.contents.Content, ak.record.Record)):
        return ak._util.wrap(layout, behavior, highlevel)
    else:
        return layout


def build_assembly(schema, container, instructions):
    if not isinstance(schema, dict):
        raise ak._errors.wrap_error(
            TypeError(f"unrecognized JSONSchema: expected dict, got {schema!r}")
        )

    if "type" not in schema is None:
        raise ak._errors.wrap_error(
            TypeError(f"unrecognized JSONSchema: no 'type' in {schema!r}")
        )

    tpe = schema["type"]

    is_optional = False
    if isinstance(tpe, list):
        if "null" in tpe:
            is_optional = True
            tpe = [x for x in tpe if x != "null"]
        if len(tpe) == 1:
            tpe = tpe[0]

    if tpe in {"boolean", "integer", "number"}:
        # https://json-schema.org/understanding-json-schema/reference/boolean.html
        # https://json-schema.org/understanding-json-schema/reference/numeric.html

        if tpe == "boolean":
            instruction = "FillBoolean"
            dtype = "uint8"
            primitive = "bool"
        elif tpe == "integer":
            instruction = "FillInteger"
            primitive = dtype = "int64"
        elif tpe == "number":
            instruction = "FillNumber"
            primitive = dtype = "float64"

        if is_optional:
            mask = f"node{len(container)}"
            container[mask + "-mask"] = None
            node = f"node{len(container)}"
            container[node + "-data"] = None
            instructions.append(["FillByteMaskedArray", mask + "-mask", "int8"])
            instructions.append([instruction, node + "-data", dtype])
            return ak.forms.ByteMaskedForm(
                "i8",
                ak.forms.NumpyForm(primitive, form_key=node),
                valid_when=True,
                form_key=mask,
            )

        else:
            node = f"node{len(container)}"
            container[node + "-data"] = None
            instructions.append([instruction, node + "-data", dtype])
            return ak.forms.NumpyForm(primitive, form_key=node)

    elif tpe == "string":
        # https://json-schema.org/understanding-json-schema/reference/string.html#string
        if "enum" in schema:
            strings = schema["enum"]
            assert isinstance(strings, list)
            assert len(strings) >= 1
            assert all(isinstance(x, str) for x in strings)
            bytestrings = [x.encode("utf-8", errors="surrogateescape") for x in strings]

            index = f"node{len(container)}"
            container[index + "-index"] = None
            offsets = f"node{len(container)}"
            container[offsets + "-offsets"] = numpy.empty(len(strings) + 1, np.int64)
            container[offsets + "-offsets"][0] = 0
            container[offsets + "-offsets"][1:] = numpy.cumsum(
                [len(x) for x in bytestrings]
            )
            node = f"container{len(container)}"
            container[node + "-data"] = b"".join(bytestrings)

            if is_optional:
                instruction = "FillNullEnumString"
                formtype = ak.forms.IndexedOptionForm
            else:
                instruction = "FillEnumString"
                formtype = ak.forms.IndexedForm

            instructions.append([instruction, index + "-index", "int64", strings])

            return formtype(
                "i64",
                ak.forms.ListOffsetForm(
                    "i64",
                    ak.forms.NumpyForm(
                        "uint8", parameters={"__array__": "char"}, form_key=node
                    ),
                    parameters={"__array__": "string"},
                    form_key=offsets,
                ),
                parameters={"__array__": "categorical"},
                form_key=index,
            )

        else:
            if is_optional:
                mask = f"node{container}"
                container[mask + "-mask"] = None
                instructions.append(["FillByteMaskedArray", mask + "-mask", "int8"])

            offsets = f"node{len(container)}"
            container[offsets + "-offsets"] = None
            node = f"node{len(container)}"
            container[node + "-data"] = None
            instructions.append(
                ["FillString", offsets + "-offsets", "int64", node + "-data", "uint8"]
            )

            out = ak.forms.ListOffsetForm(
                "i64",
                ak.forms.NumpyForm(
                    "uint8",
                    parameters={"__array__": "char"},
                    form_key=node,
                ),
                parameters={"__array__": "string"},
                form_key=offsets,
            )
            if is_optional:
                return ak.forms.ByteMaskedForm(
                    "i8", out, valid_when=True, form_key=mask
                )
            else:
                return out

    elif tpe == "array":
        # https://json-schema.org/understanding-json-schema/reference/array.html

        if "items" not in schema:
            raise ak._errors.wrap_error(
                TypeError("JSONSchema type is not concrete: array without 'items'")
            )

        if schema.get("minItems") == schema.get("maxItems") != None:  # noqa: E711
            assert ak._util.is_integer(schema.get("minItems"))

            if is_optional:
                mask = f"node{len(container)}"
                container[mask + "-index"] = None
                instructions.append(
                    ["FillIndexedOptionArray", mask + "-index", "int64"]
                )

            instructions.append(["FixedLengthList", schema.get("minItems")])

            content = build_assembly(schema["items"], container, instructions)

            out = ak.forms.RegularForm(content, size=schema.get("minItems"))
            if is_optional:
                return ak.forms.IndexedOptionForm("i64", out, form_key=mask)
            else:
                return out

        else:
            if is_optional:
                mask = f"node{len(container)}"
                container[mask + "-mask"] = None
                instructions.append(["FillByteMaskedArray", mask + "-mask", "int8"])

            offsets = f"node{len(container)}"
            container[offsets + "-offsets"] = None
            instructions.append(["VarLengthList", offsets + "-offsets", "int64"])

            content = build_assembly(schema["items"], container, instructions)

            out = ak.forms.ListOffsetForm("i64", content, form_key=offsets)
            if is_optional:
                return ak.forms.ByteMaskedForm(
                    "i8", out, valid_when=True, form_key=mask
                )
            else:
                return out

    elif tpe == "object":
        # https://json-schema.org/understanding-json-schema/reference/object.html

        if "properties" not in schema:
            raise ak._errors.wrap_error(
                TypeError(
                    "JSONSchema type is not concrete: object without 'properties'"
                )
            )

        names = []
        subschemas = []
        for name, subschema in schema["properties"].items():
            names.append(name)
            subschemas.append(subschema)

        if is_optional:
            mask = f"node{len(container)}"
            container[mask + "-index"] = None
            instructions.append(["FillIndexedOptionArray", mask + "-index", "int64"])

        instructions.append(["KeyTableHeader", len(names)])
        startkeys = len(instructions)

        for name in names:
            instructions.append(["KeyTableItem", name, None])

        contents = []
        for keyindex, subschema in enumerate(subschemas):
            # set the "jump_to" instruction position in the KeyTable
            instructions[startkeys + keyindex][2] = len(instructions)
            contents.append(build_assembly(subschema, container, instructions))

        out = ak.forms.RecordForm(contents, names)
        if is_optional:
            return ak.forms.IndexedOptionForm("i64", out, form_key=mask)
        else:
            return out

    elif isinstance(tpe, list):
        raise ak._errors.wrap_error(
            NotImplementedError("arbitrary unions of types are not yet supported")
        )

    else:
        raise ak._errors.wrap_error(TypeError(f"unrecognized JSONSchema: {tpe!r}"))
