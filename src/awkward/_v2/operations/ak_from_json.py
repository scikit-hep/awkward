# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pathlib
import json
from urllib.parse import urlparse

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


def from_json(
    source,
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
        posinf_string (None or str): If not None, strings with this value will
            be interpreted as floating-point positive infinity values.
        neginf_string (None or str): If not None, strings with this value
            will be interpreted as floating-point negative infinity values.
        complex_record_fields (None or (str, str)): If not None, defines a pair of
            field names to interpret 2-field records as complex numbers.
        buffersize (int): Number of bytes in each read from source: larger
            values use more memory but read less frequently. (Python GIL is
            released before and after read events.)
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

    Supported JSONSchema elements:

      * The root of the schema must be `"type": "array"` or `"type": "object"`.
      * Every level must have a `"type"`, which can only name one type (as a string
        or length-1 list) or one type and `"null"` (as a length-2 list).
      * `"type": "boolean"` \u2192 1-byte boolean values.
      * `"type": "integer"` \u2192 8-byte integer values.
        Numbers may include a fractional part, as per the JSONSchema specification,
        but this function ignores any fractional part.
      * `"type": "number"` \u2192 8-byte floating-point values.
      * `"type": "string"` \u2192 UTF-8 encoded strings. All JSON escape sequences are
        supported. Remember that the `source` data are ASCII; Unicode is derived from
        "`\\uXXXX`" escape sequences. If an `"enum"` is given, strings are represented
        as categorical values (#ak.layout.IndexedArray or #ak.layout.IndexedOptionArray).
      * `"type": "array"` \u2192 nested lists. The `"items"` must be specified. If
        `"minItems"` and `"maxItems"` are specified and equal to each other, the
        list has regular-type (#ak.types.RegularType); otherwise, it has variable-length
        type (#ak.types.ListType).
      * `"type": "object"` \u2192 nested records. The `"properties"` must be specified,
        and any properties in the data not described by `"properties"` will not
        appear in the output.

    See also #ak.to_json.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.from_json",
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
        isinstance(complex_record_fields, tuple)
        and len(complex_record_fields) == 2
        and isinstance(complex_record_fields[0], str)
        and isinstance(complex_record_fields[1], str)
    ):

        def action(node, **kwargs):
            if isinstance(node, ak._v2.contents.RecordArray):
                if set(complex_record_fields).issubset(node.fields):
                    real = node._getitem_field(complex_record_fields[0])
                    imag = node._getitem_field(complex_record_fields[1])
                    if (
                        isinstance(real, ak._v2.contents.NumpyArray)
                        and len(real.shape) == 1
                        and issubclass(real.dtype.type, (np.integer, np.floating))
                        and isinstance(imag, ak._v2.contents.NumpyArray)
                        and len(imag.shape) == 1
                        and issubclass(imag.dtype.type, (np.integer, np.floating))
                    ):
                        return ak._v2.contents.NumpyArray(
                            node._nplike.asarray(real) + node._nplike.asarray(imag) * 1j
                        )
                    else:
                        raise ak._v2._util.error(
                            ValueError(
                                f"expected record with fields {complex_record_fields[0]!r} and {complex_record_fields[1]!r} to have integer or floating point types, not {str(real.form.type)!r} and {str(imag.form.type)!r}"
                            )
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
    posinf_string,
    neginf_string,
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
        try:
            ak._ext.fromjsonobj(
                obj,
                builder,
                read_one,
                buffersize,
                nan_string,
                posinf_string,
                neginf_string,
            )
        except Exception as err:
            raise ak._v2._util.error(ValueError(str(err))) from None

    formstr, length, buffers = builder.to_buffers()
    form = ak._v2.forms.from_json(formstr)
    layout = ak._v2.operations.from_buffers(form, length, buffers, highlevel=False)

    layout = _record_to_complex(layout, complex_record_fields)

    if read_one:
        layout = layout[0]

    if highlevel and isinstance(
        layout, (ak._v2.contents.Content, ak._v2.record.Record)
    ):
        return ak._v2._util.wrap(layout, behavior, highlevel)
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
    if isinstance(schema, bytes) or ak._v2._util.isstr(schema):
        schema = json.loads(schema)

    if not isinstance(schema, dict):
        raise ak._v2._util.error(
            TypeError(f"unrecognized JSONSchema: expected dict, got {schema!r}")
        )

    container = {}
    instructions = []

    if schema.get("type") == "array":
        if "items" not in schema:
            raise ak._v2._util.error(
                TypeError("JSONSchema type is not concrete: array without items")
            )

        instructions.append(["TopLevelArray"])
        form = build_assembly(schema["items"], container, instructions)

    elif schema.get("type") == "object":
        form = build_assembly(schema, container, instructions)

    else:
        raise ak._v2._util.error(
            TypeError(
                "only 'array' and 'object' types supported at the JSONSchema root"
            )
        )

    read_one = not line_delimited

    with _get_reader(source)() as obj:
        try:
            length = ak._ext.fromjsonobj_schema(
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
            raise ak._v2._util.error(ValueError(str(err))) from None

    layout = ak._v2.operations.from_buffers(form, length, container, highlevel=False)
    layout = _record_to_complex(layout, complex_record_fields)

    if highlevel and isinstance(
        layout, (ak._v2.contents.Content, ak._v2.record.Record)
    ):
        return ak._v2._util.wrap(layout, behavior, highlevel)
    else:
        return layout


def build_assembly(schema, container, instructions):
    if not isinstance(schema, dict):
        raise ak._v2._util.error(
            TypeError(f"unrecognized JSONSchema: expected dict, got {schema!r}")
        )

    if "type" not in schema is None:
        raise ak._v2._util.error(
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
            return ak._v2.forms.ByteMaskedForm(
                "i8",
                ak._v2.forms.NumpyForm(primitive, form_key=node),
                valid_when=True,
                form_key=mask,
            )

        else:
            node = f"node{len(container)}"
            container[node + "-data"] = None
            instructions.append([instruction, node + "-data", dtype])
            return ak._v2.forms.NumpyForm(primitive, form_key=node)

    elif tpe == "string":
        # https://json-schema.org/understanding-json-schema/reference/string.html#string
        if "enum" in schema:
            strings = schema["enum"]
            assert isinstance(strings, list)
            assert len(strings) >= 1
            assert all(ak._v2._util.isstr(x) for x in strings)
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
                formtype = ak._v2.forms.IndexedOptionForm
            else:
                instruction = "FillEnumString"
                formtype = ak._v2.forms.IndexedForm

            instructions.append([instruction, index + "-index", "int64", strings])

            return formtype(
                "i64",
                ak._v2.forms.ListOffsetForm(
                    "i64",
                    ak._v2.forms.NumpyForm(
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

            out = ak._v2.forms.ListOffsetForm(
                "i64",
                ak._v2.forms.NumpyForm(
                    "uint8",
                    parameters={"__array__": "char"},
                    form_key=node,
                ),
                parameters={"__array__": "string"},
                form_key=offsets,
            )
            if is_optional:
                return ak._v2.forms.ByteMaskedForm(
                    "i8", out, valid_when=True, form_key=mask
                )
            else:
                return out

    elif tpe == "array":
        # https://json-schema.org/understanding-json-schema/reference/array.html

        if "items" not in schema:
            raise ak._v2._util.error(
                TypeError("JSONSchema type is not concrete: array without 'items'")
            )

        if schema.get("minItems") == schema.get("maxItems") != None:  # noqa: E711
            assert ak._v2._util.isint(schema.get("minItems"))

            if is_optional:
                mask = f"node{len(container)}"
                container[mask + "-index"] = None
                instructions.append(
                    ["FillIndexedOptionArray", mask + "-index", "int64"]
                )

            instructions.append(["FixedLengthList", schema.get("minItems")])

            content = build_assembly(schema["items"], container, instructions)

            out = ak._v2.forms.RegularForm(content, size=schema.get("minItems"))
            if is_optional:
                return ak._v2.forms.IndexedOptionForm("i64", out, form_key=mask)
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

            out = ak._v2.forms.ListOffsetForm("i64", content, form_key=offsets)
            if is_optional:
                return ak._v2.forms.ByteMaskedForm(
                    "i8", out, valid_when=True, form_key=mask
                )
            else:
                return out

    elif tpe == "object":
        # https://json-schema.org/understanding-json-schema/reference/object.html

        if "properties" not in schema:
            raise ak._v2._util.error(
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

        out = ak._v2.forms.RecordForm(contents, names)
        if is_optional:
            return ak._v2.forms.IndexedOptionForm("i64", out, form_key=mask)
        else:
            return out

    elif isinstance(tpe, list):
        raise ak._v2._util.error(
            NotImplementedError("arbitrary unions of types are not yet supported")
        )

    else:
        raise ak._v2._util.error(TypeError(f"unrecognized JSONSchema: {tpe!r}"))
