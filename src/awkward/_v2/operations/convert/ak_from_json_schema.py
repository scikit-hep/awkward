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
    initial=1024,
    resize=1.5,
):
    if isinstance(source, bytes):
        pass
    elif ak._v2._util.isstr(source):
        source = source.encode("ascii")
    else:
        raise NotImplementedError("for now, 'source' must be bytes or str")

    if ak._v2._util.isstr(schema):
        schema = json.loads(schema)

    if not isinstance(schema, dict):
        raise TypeError(
            "malformed jsonschema: expected dict, got {0}".format(repr(schema))
        )

    if schema.get("type") == "array":
        if "items" not in schema:
            raise TypeError("jsonschema type is not concrete: array without items")

        outputs = {}
        initialization = []
        instructions = []

        form = build_forth(
            schema["items"], outputs, initialization, instructions, "  ", bits64
        )

        forthcode = r"""input source
{0}
{1}

source skipws
source enumonly s" [" drop
source skipws
0
source enum s" ]"
begin
while
  source skipws
{2}
  1+
  source skipws
  source enumonly s" ]" s" ,"
repeat
""".format(
            "\n".join("output {0} {1}".format(n, t) for n, t in outputs.items()),
            "\n".join(initialization),
            "\n".join(instructions),
        )

        # print(form)
        # print(forthcode)

        if bits64:
            vm = ForthMachine64(forthcode)
        else:
            vm = ForthMachine32(forthcode)

        try:
            vm.run({"source": source})

        except ValueError:
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
            "unrecognized jsonschema: expected dict, got {0}".format(repr(schema))
        )

    if "type" not in schema is None:
        raise TypeError(
            "unrecognized jsonschema: no 'type' in {0}".format(repr(schema))
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
            raise TypeError("jsonschema type is not concrete: array without 'items'")

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
                    r"""{0}    source skipws""".format(indent),
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
                    r"""{0}  source skipws""".format(indent),
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
                "jsonschema type is not concrete: object without 'properties'"
            )

        if "required" not in schema or set(schema["required"]) != set(
            schema["properties"]
        ):
            raise TypeError(
                "jsonschema type is not concrete: object's 'required' must include all 'properties'"
            )

        names = []
        subschemas = []
        for k, v in schema["properties"].items():
            names.append(k)
            subschemas.append(v)

        instructions.extend(
            [
                r"""{0}source enumonly s" {{" drop""".format(indent),
                r"""{0}source skipws""".format(indent),
                r"""{0}source enum s" }}" """.format(indent),
                r"""{0}begin""".format(indent),
                r"""{0}while""".format(indent),
                r"""{0}  source skipws""".format(indent),
            ]
        )
        # contents = []

        instructions.extend(
            [
                r"""{0}""".format(indent),
                r"""{0}""".format(indent),
                r"""{0}""".format(indent),
                r"""{0}""".format(indent),
                r"""{0}""".format(indent),
            ]
        )

        raise NotImplementedError

    elif isinstance(tpe, list):
        raise NotImplementedError("arbitrary unions of types are not yet supported")

    else:
        raise TypeError("unrecognized jsonschema: {0}".format(repr(tpe)))
