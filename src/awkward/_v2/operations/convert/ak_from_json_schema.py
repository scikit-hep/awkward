# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json
import os

import awkward as ak
from awkward.forth import ForthMachine64

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


def from_json_schema(
    source,
    schema,
    highlevel=True,
    behavior=None,
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
        instructions = []

        form = build_forth(schema["items"], outputs, instructions, "  ")

        forthcode = r"""input source
{0}

source skipws
source enumonly s" [" drop
source skipws

0
source enum s" ]"
begin
  0= invert
while
  source skipws
{1}

  1+
  source skipws
  source enumonly s" ]" s" ,"
repeat
""".format(
            " ".join("output {0} {1}".format(n, t) for n, t in outputs.items()),
            "\n".join(instructions),
        )

        # print(form)
        # print(forthcode)

        vm = ForthMachine64(forthcode)
        try:
            vm.run({"source": source})

        except ValueError:
            position = vm.input_position("source")
            before = source[position - 30 : position].decode(
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


def build_forth(schema, outputs, instructions, indent):
    if not isinstance(schema, dict):
        raise TypeError(
            "unrecognized jsonschema: expected dict, got {0}".format(repr(schema))
        )

    if "type" not in schema is None:
        raise TypeError(
            "unrecognized jsonschema: no 'type' in {0}".format(repr(schema))
        )

    tpe = schema["type"]

    # is_option = False
    # if isinstance(tpe, list):
    #     if "null" in tpe:
    #         is_option = True
    #         tpe = [x for x in tpe if x != "null"]
    #         if len(tpe) == 1:
    #             tpe = tpe[0]

    if tpe == "boolean":
        raise NotImplementedError

    elif tpe == "integer":
        # https://json-schema.org/understanding-json-schema/reference/numeric.html
        node = "node{0}".format(len(outputs))
        outputs[node + "-data"] = "int64"
        instructions.extend(
            [
                "",
                # note: accepts (and truncates) fractional part
                "{0}source textfloat-> {1}-data".format(indent, node),
            ]
        )
        return ak._v2.forms.NumpyForm("int64", form_key=node)

    elif tpe == "number":
        raise NotImplementedError

    elif tpe == "string":
        raise NotImplementedError

    elif tpe == "array":
        raise NotImplementedError

    elif tpe == "object":
        raise NotImplementedError

    elif isinstance(tpe, list):
        raise NotImplementedError

    else:
        raise TypeError("unrecognized jsonschema: {0}".format(repr(tpe)))
