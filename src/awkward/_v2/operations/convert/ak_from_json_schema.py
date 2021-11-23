# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json
import os

import awkward as ak
from awkward.forth import ForthMachine32, ForthMachine64

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


def from_json_schema(
    source,
    schema,
    highlevel=True,
    behavior=None,
    bits64=True,
    stack_size=1024,
    recursion_depth=1024,
    string_buffer_size=1024,
    output_initial_size=1024,
    output_resize_factor=1.5,
):
    u"""
    Args:
        source (str or bytes): JSON-formatted string to convert into an array.
            Only ASCII bytes are supported; for Unicode in the output data, use
            JSON's "`\\uXXXX`" escape sequences in the source.
        schema (str, bytes, or nested dicts): JSONSchema to assume in the parsing.
            The JSON data are *not* validated against the schema; the schema is
            only used to accelerate parsing.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        bits64 (bool): If True, use a 64-bit Forth machine and generate 64-bit
            offsets and indexes for structure; otherwise, use 32-bits. Regardless
            of this setting, numerical values from the JSON text are represented
            as int64 integers and float64 floating-point numbers.
        stack_size (int): Fixed depth of the Forth data stack. If you're getting
            "stack overflow" errors, increase this parameter.
        recursion_depth (int): Fixed depth of the Forth instruction stack. If you're
            getting "recursion depth exceeded" errors, increase this parameter.
        string_buffer_size (int): Fixed size of readable strings in bytes. If your JSON
            contains large string values and it is declared invalid at the position
            of such a string, increase this value.
        output_initial_size (int): Initial size (in bytes) of output buffers, which
            grow as needed to accommodate the size of the dataset.
        output_resize_factor (float): Resize multiplier for output buffers, which
            determines how quickly they grow; should be strictly greater than 1.

    Converts a JSON string into an Awkward Array, using a JSONSchema to accelerate
    the parsing of the source and building of the output. The JSON data are not
    *validated* against the schema; the schema is *assumed* to be correct.

    Supported JSONSchema elements:

      * The root of the schema must be `"type": "array"` or `"type": "object"`.
      * Every level must have a `"type"`, which can only name one type or
        one type and `"null"`.
      * `"type": "boolean"` \u2192 1-byte boolean values.
      * `"type": "integer"` \u2192 8-byte integer values (regardless of `bits64`).
        Numbers may include a fractional part, as per the JSONSchema specification,
        but this function ignores any fractional part.
      * `"type": "number"` \u2192 8-byte floating-point values (regardless of `bits64`).
      * `"type": "string"` \u2192 UTF-8 encoded strings. All JSON escape sequences are
        supported. Remember that the `source` data are ASCII; Unicode is derived from
        "`\\uXXXX`" escape sequences. If an `"enum"` is given, strings are represented
        as categorical values (#ak.layout.IndexedArray).
      * `"type": "array"` \u2192 nested lists. The `"items"` must be specified. If
        `"minItems"` and `"maxItems"` are specified and equal to each other, the
        list has regular-type (#ak.types.RegularType); otherwise, it has variable-length
        type (#ak.types.ListType).
      * `"type": "object"` \u2192 nested records. The `"properties"` must be specified,
        and `"required"` must be a list of all property names.

    Internally, this function uses the schema to generate a specialized Forth machine
    that parses only the JSON type that the schema specifies, avoiding unnecessary
    type checks. It also avoids the type-discovery of #ak.ArrayBuilder for further
    speedup. Most of the arguments (`bits64`, `stack_size`, `recursion_depth`,
    `string_buffer_size`, `output_initial_size`, and `output_resize_factor`) configure
    the Forth machine.

    See also #ak.from_json and #ak.to_json.
    """
    if isinstance(source, bytes):
        pass
    elif ak._v2._util.isstr(source):
        source = source.encode("ascii")
    else:
        raise NotImplementedError("for now, 'source' must be bytes or str")

    if isinstance(schema, bytes) or ak._v2._util.isstr(schema):
        schema = json.loads(schema)

    if not isinstance(schema, dict):
        raise TypeError(
            "malformed JSONSchema: expected dict, got {0}".format(repr(schema))
        )

    if schema.get("type") == "array":
        if "items" not in schema:
            raise TypeError("JSONSchema type is not concrete: array without items")

        outputs = {}
        initialization = []
        instructions = []

        form = build_forth(
            schema["items"], outputs, initialization, instructions, "  ", bits64
        )

        forthcode = r"""input source
{0}{1}

source skipws
source enumonly s" [" drop
source skipws
0
source enum s" ]"
begin
while
{2}
  1+
  source skipws
  source enumonly s" ]" s" ,"
  source skipws
repeat
source skipws
""".format(
            "\n".join("output {0} {1}".format(n, t) for n, t in outputs.items()),
            "".join("\n" + x for x in initialization),
            "\n".join(instructions),
        )

        options = {
            "stack_size": stack_size,
            "recursion_depth": recursion_depth,
            "string_buffer_size": string_buffer_size,
            "output_initial_size": output_initial_size,
            "output_resize_factor": output_resize_factor,
        }
        if bits64:
            vm = ForthMachine64(forthcode, **options)
        else:
            vm = ForthMachine32(forthcode, **options)

        try:
            vm.run({"source": source})

            if vm.input_position("source") != len(source):
                raise ValueError

        except ValueError as err:
            if (
                "read beyond" in str(err)
                or "varint too big" in str(err)
                or "text number missing" in str(err)
                or "quoted string missing" in str(err)
                or "enumeration missing" in str(err)
            ):
                position = vm.input_position("source")
                before = source[max(0, position - 30) : position].decode(
                    "ascii", errors="surrogateescape"
                )
                before = before.replace(os.linesep, repr(os.linesep).strip("'\""))
                if position - 30 > 0:
                    before = "..." + before
                after = source[position : position + 30].decode(
                    "ascii", errors="surrogateescape"
                )
                if position + 30 < len(source):
                    after = after + "..."
                raise ValueError(
                    "JSON is invalid or does not fit schema at position {0}:\n\n    {1}\n    {2}".format(
                        position, before + after, "-" * len(before) + "^"
                    )
                )
            else:
                raise err

        (length,) = vm.stack
        contents = {}
        for name in outputs:
            contents[name] = numpy.asarray(vm[name])

        return ak._v2.operations.convert.from_buffers(form, length, contents)

    elif schema.get("type") == "object":
        raise NotImplementedError

    else:
        raise TypeError("only 'array' and 'object' types supported at schema top-level")


def build_forth(schema, outputs, initialization, instructions, indent, bits64):
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

    if tpe == "boolean":
        # https://json-schema.org/understanding-json-schema/reference/boolean.html
        if is_optional:
            mask = "node{0}".format(len(outputs))
            outputs[mask + "-mask"] = "int8"
            node = "node{0}".format(len(outputs))
            outputs[node + "-data"] = "int8"
            instructions.extend(
                [
                    r"""{0}source enumonly s" null" s" false" s" true" dup""".format(
                        indent
                    ),
                    r"""{0}1- {1}-data <- stack""".format(indent, node),
                    r"""{0}{1}-mask <- stack""".format(indent, mask),
                ]
            )
            return ak._v2.forms.ByteMaskedForm(
                "i8",
                ak._v2.forms.NumpyForm(outputs[node + "-data"], form_key=node),
                valid_when=True,
                form_key=mask,
            )

        else:
            node = "node{0}".format(len(outputs))
            outputs[node + "-data"] = "int8"
            instructions.extend(
                [
                    """{0}source enum s" false" s" true" {1}-data <- stack""".format(
                        indent, node
                    ),
                ]
            )
            return ak._v2.forms.NumpyForm(outputs[node + "-data"], form_key=node)

    elif tpe == "integer" or tpe == "number":
        # https://json-schema.org/understanding-json-schema/reference/numeric.html
        if is_optional:
            mask = "node{0}".format(len(outputs))
            outputs[mask + "-mask"] = "int8"
            node = "node{0}".format(len(outputs))
            if tpe == "integer":
                outputs[node + "-data"] = "int64"
            else:
                outputs[node + "-data"] = "float64"
            instructions.extend(
                [
                    r"""{0}source enum s" null" dup if""".format(indent),
                    # note: we want textfloat-> for integers, too
                    r"""{0}  source textfloat-> {1}-data""".format(indent, node),
                    r"""{0}else""".format(indent),
                    r"""{0}  0 {1}-data <- stack""".format(indent, node),
                    r"""{0}then""".format(indent),
                    r"""{0}{1}-mask <- stack""".format(indent, mask),
                ]
            )
            return ak._v2.forms.ByteMaskedForm(
                "i8",
                ak._v2.forms.NumpyForm(outputs[node + "-data"], form_key=node),
                valid_when=True,
                form_key=mask,
            )

        else:
            node = "node{0}".format(len(outputs))
            if tpe == "integer":
                outputs[node + "-data"] = "int64"
            else:
                outputs[node + "-data"] = "float64"
            instructions.extend(
                [
                    # note: we want textfloat-> for integers, too
                    r"""{0}source textfloat-> {1}-data""".format(indent, node),
                ]
            )
            return ak._v2.forms.NumpyForm(outputs[node + "-data"], form_key=node)

    elif tpe == "string":
        # https://json-schema.org/understanding-json-schema/reference/string.html#string
        if is_optional:
            mask = "node{0}".format(len(outputs))
            outputs[mask + "-mask"] = "int8"
            offsets = "node{0}".format(len(outputs))
            outputs[offsets + "-offsets"] = "int64" if bits64 else "int32"
            node = "node{0}".format(len(outputs))
            outputs[node + "-data"] = "uint8"
            initialization.append(r"""0 {0}-offsets <- stack""".format(offsets))
            instructions.extend(
                [
                    r"""{0}source enum s" null" dup if""".format(indent),
                    r"""{0}  source quotedstr-> {1}-data""".format(indent, node),
                    r"""{0}  {1}-offsets +<- stack""".format(indent, offsets),
                    r"""{0}else""".format(indent),
                    r"""{0}  0 {1}-offsets +<- stack""".format(indent, offsets),
                    r"""{0}then""".format(indent),
                    r"""{0}{1}-mask <- stack""".format(indent, mask),
                ]
            )
            return ak._v2.forms.ByteMaskedForm(
                "i8",
                ak._v2.forms.ListOffsetForm(
                    "i64" if bits64 else "i32",
                    ak._v2.forms.NumpyForm(
                        outputs[node + "-data"],
                        parameters={"__array__": "char"},
                        form_key=node,
                    ),
                    parameters={"__array__": "string"},
                    form_key=offsets,
                ),
                valid_when=True,
                form_key=mask,
            )

        else:
            offsets = "node{0}".format(len(outputs))
            outputs[offsets + "-offsets"] = "int64" if bits64 else "int32"
            node = "node{0}".format(len(outputs))
            outputs[node + "-data"] = "uint8"
            initialization.append(r"""0 {0}-offsets <- stack""".format(offsets))
            instructions.extend(
                [
                    r"""{0}source quotedstr-> {1}-data""".format(indent, node),
                    r"""{0}{1}-offsets +<- stack""".format(indent, offsets),
                ]
            )
            return ak._v2.forms.ListOffsetForm(
                "i64" if bits64 else "i32",
                ak._v2.forms.NumpyForm(
                    outputs[node + "-data"],
                    parameters={"__array__": "char"},
                    form_key=node,
                ),
                parameters={"__array__": "string"},
                form_key=offsets,
            )

    elif tpe == "array":
        # https://json-schema.org/understanding-json-schema/reference/array.html

        if "items" not in schema:
            raise TypeError("JSONSchema type is not concrete: array without 'items'")

        if schema.get("minItems") == schema.get("maxItems") != None:  # noqa: E711
            if is_optional:
                mask = "node{0}".format(len(outputs))
                outputs[mask + "-index"] = "int64" if bits64 else "int32"
                initialization.append("variable {0}-count".format(mask))
                instructions.extend(
                    [
                        r"""{0}source enumonly s" null" s" [" if""".format(indent),
                        r"""{0}  {1}-count @ {1}-index <- stack""".format(indent, mask),
                        r"""{0}  1 {1}-count +!""".format(indent, mask),
                    ]
                )
                indent = indent + "  "
            else:
                instructions.append(r"""{0}source enumonly s" [" drop""".format(indent))

            instructions.extend(
                [
                    r"""{0}source skipws""".format(indent),
                    r"""{0}source enum s" ]" """.format(indent),
                    r"""{0}begin""".format(indent),
                    r"""{0}while""".format(indent),
                ]
            )
            content = build_forth(
                schema["items"],
                outputs,
                initialization,
                instructions,
                indent + "  ",
                bits64,
            )
            instructions.extend(
                [
                    r"""{0}  source skipws""".format(indent),
                    r"""{0}  source enumonly s" ]" s" ," """.format(indent),
                    r"""{0}  source skipws""".format(indent),
                    r"""{0}repeat""".format(indent),
                ]
            )

            if is_optional:
                indent = indent[:-2]
                instructions.extend(
                    [
                        r"""{0}else""".format(indent),
                        r"""{0}  -1 {1}-index <- stack""".format(indent, mask),
                        r"""{0}then""".format(indent),
                    ]
                )
                return ak._v2.forms.IndexedOptionForm(
                    "i64" if bits64 else "i32",
                    ak._v2.forms.RegularForm(content, size=schema.get("minItems")),
                    form_key=mask,
                )

            else:
                return ak._v2.forms.RegularForm(content, size=schema.get("minItems"))

        else:
            if is_optional:
                mask = "node{0}".format(len(outputs))
                outputs[mask + "-mask"] = "int8"
                offsets = "node{0}".format(len(outputs))
                outputs[offsets + "-offsets"] = "int64" if bits64 else "int32"
                initialization.append(r"""0 {0}-offsets <- stack""".format(offsets))
                instructions.extend(
                    [
                        r"""{0}source enumonly s" null" s" [" dup if""".format(indent),
                        r"""{0}  source skipws""".format(indent),
                        r"""{0}  0""".format(indent),
                        r"""{0}  source enum s" ]" """.format(indent),
                        r"""{0}  begin""".format(indent),
                        r"""{0}  while""".format(indent),
                    ]
                )
                content = build_forth(
                    schema["items"],
                    outputs,
                    initialization,
                    instructions,
                    indent + "    ",
                    bits64,
                )
                instructions.extend(
                    [
                        r"""{0}    1+""".format(indent),
                        r"""{0}    source skipws""".format(indent),
                        r"""{0}    source enumonly s" ]" s" ," """.format(indent),
                        r"""{0}    source skipws""".format(indent),
                        r"""{0}  repeat""".format(indent),
                        r"""{0}  {1}-offsets +<- stack""".format(indent, offsets),
                        r"""{0}else""".format(indent),
                        r"""{0}  0 {1}-offsets +<- stack""".format(indent, offsets),
                        r"""{0}then""".format(indent),
                        r"""{0}{1}-mask <- stack""".format(indent, mask),
                    ]
                )
                return ak._v2.forms.ByteMaskedForm(
                    "i8",
                    ak._v2.forms.ListOffsetForm(
                        "i64" if bits64 else "i32", content, form_key=offsets
                    ),
                    valid_when=True,
                    form_key=mask,
                )

            else:
                offsets = "node{0}".format(len(outputs))
                outputs[offsets + "-offsets"] = "int64" if bits64 else "int32"
                initialization.append(r"""0 {0}-offsets <- stack""".format(offsets))
                instructions.extend(
                    [
                        r"""{0}source enumonly s" [" drop""".format(indent),
                        r"""{0}source skipws""".format(indent),
                        r"""{0}0""".format(indent),
                        r"""{0}source enum s" ]" """.format(indent),
                        r"""{0}begin""".format(indent),
                        r"""{0}while""".format(indent),
                    ]
                )
                content = build_forth(
                    schema["items"],
                    outputs,
                    initialization,
                    instructions,
                    indent + "  ",
                    bits64,
                )
                instructions.extend(
                    [
                        r"""{0}  1+""".format(indent),
                        r"""{0}  source skipws""".format(indent),
                        r"""{0}  source enumonly s" ]" s" ," """.format(indent),
                        r"""{0}  source skipws""".format(indent),
                        r"""{0}repeat""".format(indent),
                        r"""{0}{1}-offsets +<- stack""".format(indent, offsets),
                    ]
                )
                return ak._v2.forms.ListOffsetForm(
                    "i64" if bits64 else "i32", content, form_key=offsets
                )

    elif tpe == "object":
        # https://json-schema.org/understanding-json-schema/reference/object.html

        if "properties" not in schema:
            raise TypeError(
                "JSONSchema type is not concrete: object without 'properties'"
            )

        if "required" not in schema or set(schema["required"]) != set(
            schema["properties"]
        ):
            raise TypeError(
                "JSONSchema type is not concrete: object's 'required' must include all 'properties'"
            )

        names = []
        subschemas = []
        for name, subschema in schema["properties"].items():
            names.append(name)
            subschemas.append(subschema)

        if is_optional:
            mask = "node{0}".format(len(outputs))
            outputs[mask + "-index"] = "int64" if bits64 else "int32"
            initialization.append("variable {0}-count".format(mask))
            instructions.extend(
                [
                    r"""{0}source enumonly s" null" s" {{" if""".format(indent),
                    r"""{0}  {1}-count @ {1}-index <- stack""".format(indent, mask),
                    r"""{0}  1 {1}-count +!""".format(indent, mask),
                ]
            )
            indent = indent + "  "
        else:
            instructions.append(r"""{0}source enumonly s" {{" drop""".format(indent))

        instructions.extend(
            [
                r"""{0}source skipws""".format(indent),
                r"""{0}source enum s" }}" """.format(indent),
                r"""{0}begin""".format(indent),
                r"""{0}while""".format(indent),
                r"""{0}  source enumonly {1}""".format(
                    indent,
                    " ".join(r's" \"' + json.dumps(x)[1:-1] + r'\""' for x in names),
                ),
                r"""{0}  source skipws""".format(indent),
                r"""{0}  source enumonly s" :" drop """.format(indent),
                r"""{0}  source skipws""".format(indent),
                r"""{0}  case""".format(indent),
            ]
        )

        contents = []
        for index, subschema in enumerate(subschemas):
            instructions.append(r"""{0}    {1} of""".format(indent, index))
            contents.append(
                build_forth(
                    subschema,
                    outputs,
                    initialization,
                    instructions,
                    indent + "      ",
                    bits64,
                )
            )
            instructions.append(r"""{0}    endof""".format(indent))

        instructions.extend(
            [
                r"""{0}  endcase""".format(indent),
                r"""{0}  source skipws""".format(indent),
                r"""{0}  source enumonly s" }}" s" ," """.format(indent),
                r"""{0}  source skipws""".format(indent),
                r"""{0}repeat""".format(indent),
            ]
        )

        if is_optional:
            indent = indent[:-2]
            instructions.extend(
                [
                    r"""{0}else""".format(indent),
                    r"""{0}  -1 {1}-index <- stack""".format(indent, mask),
                    r"""{0}then""".format(indent),
                ]
            )
            return ak._v2.forms.IndexedOptionForm(
                "i64" if bits64 else "i32",
                ak._v2.forms.RecordForm(contents, names),
                form_key=mask,
            )

        else:
            return ak._v2.forms.RecordForm(contents, names)

    elif isinstance(tpe, list):
        raise NotImplementedError("arbitrary unions of types are not yet supported")

    else:
        raise TypeError("unrecognized JSONSchema: {0}".format(repr(tpe)))
