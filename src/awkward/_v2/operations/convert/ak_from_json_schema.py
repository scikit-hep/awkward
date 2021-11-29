# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json
import os

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


def from_json_schema(
    source,
    schema,
    highlevel=True,
    behavior=None,
    output_initial_size=1024,
    output_resize_factor=1.5,
):
    u"""
    Args:
        source (str or bytes): JSON-formatted string to convert into an array.
        schema (str, bytes, or nested dicts): JSONSchema to assume in the parsing.
            The JSON data are *not* validated against the schema; the schema is
            only used to accelerate parsing.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        output_initial_size (int): Initial size (in bytes) of output buffers, which
            grow as needed to accommodate the size of the dataset.
        output_resize_factor (float): Resize multiplier for output buffers, which
            determines how quickly they grow; should be strictly greater than 1.

    Converts a JSON string into an Awkward Array, using a JSONSchema to accelerate
    the parsing of the source and building of the output. The JSON data are not
    *validated* against the schema; the schema is *assumed* to be correct.

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

    See also #ak.from_json and #ak.to_json.
    """
    if not isinstance(source, bytes) and not ak._v2._util.isstr(source):
        raise NotImplementedError("for now, 'source' must be bytes or str")

    if isinstance(schema, bytes) or ak._v2._util.isstr(schema):
        schema = json.loads(schema)

    if not isinstance(schema, dict):
        raise TypeError(
            "malformed JSONSchema: expected dict, got {0}".format(repr(schema))
        )

    container = {}
    instructions = []

    if schema.get("type") == "array":
        if "items" not in schema:
            raise TypeError("JSONSchema type is not concrete: array without items")

        instructions.append(["TopLevelArray"])
        form = build_assembly(schema["items"], container, instructions)

    elif schema.get("type") == "object":
        form = build_assembly(schema, container, instructions)

    else:
        raise TypeError(
            "only 'array' and 'object' types supported at the JSONSchema root"
        )

    specializedjson = ak._ext.SpecializedJSON(
        json.dumps(instructions),
        output_initial_size=output_initial_size,
        output_resize_factor=output_resize_factor,
    )

    if not specializedjson.parse_string(source):
        position = specializedjson.json_position
        before = source[max(0, position - 30) : position]
        if isinstance(before, bytes):
            before = before.decode("ascii", errors="surrogateescape")
        before = before.replace(os.linesep, repr(os.linesep).strip("'\""))
        if position - 30 > 0:
            before = "..." + before
        after = source[position : position + 30]
        if isinstance(after, bytes):
            after = after.decode("ascii", errors="surrogateescape")
        if position + 30 < len(source):
            after = after + "..."
        raise ValueError(
            "JSON is invalid or does not fit schema at position {0}:\n\n    {1}\n    {2}".format(
                position, before + after, "-" * len(before) + "^"
            )
        )

    for key, value in container.items():
        if value is None:
            container[key] = specializedjson[key]

    if schema.get("type") == "array":
        length = len(specializedjson)
    else:
        length = 1

    out = ak._v2.operations.convert.from_buffers(form, length, container)

    if schema.get("type") == "array":
        return out
    else:
        return out[0]


def build_assembly(schema, container, instructions):
    if not isinstance(schema, dict):
        raise TypeError(
            "unrecognized JSONSchema: expected dict, got {0}".format(repr(schema))
        )

    if "type" not in schema is None:
        raise TypeError(
            "unrecognized JSONSchema: no 'type' in {0}".format(repr(schema))
        )

    tpe = schema["type"]

    is_optional = False
    if isinstance(tpe, list):
        if "null" in tpe:
            is_optional = True
            tpe = [x for x in tpe if x != "null"]
        if len(tpe) == 1:
            tpe = tpe[0]

    if tpe == "boolean" or tpe == "integer" or tpe == "number":
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
            mask = "node{0}".format(len(container))
            container[mask + "-mask"] = None
            node = "node{0}".format(len(container))
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
            node = "node{0}".format(len(container))
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

            index = "node{0}".format(len(container))
            container[index + "-index"] = None
            offsets = "node{0}".format(len(container))
            container[offsets + "-offsets"] = numpy.empty(len(strings) + 1, np.int64)
            container[offsets + "-offsets"][0] = 0
            container[offsets + "-offsets"][1:] = numpy.cumsum(
                [len(x) for x in bytestrings]
            )
            node = "container{0}".format(len(container))
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
                mask = "node{0}".format(container)
                container[mask + "-mask"] = None
                instructions.append(["FillByteMaskedArray", mask + "-mask", "int8"])

            offsets = "node{0}".format(len(container))
            container[offsets + "-offsets"] = None
            node = "node{0}".format(len(container))
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
            raise TypeError("JSONSchema type is not concrete: array without 'items'")

        if schema.get("minItems") == schema.get("maxItems") != None:  # noqa: E711
            assert ak._v2._util.isint(schema.get("minItems"))

            if is_optional:
                mask = "node{0}".format(len(container))
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
                mask = "node{0}".format(len(container))
                container[mask + "-mask"] = None
                instructions.append(["FillByteMaskedArray", mask + "-mask", "int8"])

            offsets = "node{0}".format(len(container))
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
            raise TypeError(
                "JSONSchema type is not concrete: object without 'properties'"
            )

        names = []
        subschemas = []
        for name, subschema in schema["properties"].items():
            names.append(name)
            subschemas.append(subschema)

        if is_optional:
            mask = "node{0}".format(len(container))
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
        raise NotImplementedError("arbitrary unions of types are not yet supported")

    else:
        raise TypeError("unrecognized JSONSchema: {0}".format(repr(tpe)))
