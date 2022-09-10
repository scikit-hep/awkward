import json

import lark

import awkward as ak

from awkward.types.numpytype import NumpyType
from awkward.types.unknowntype import UnknownType
from awkward.types.regulartype import RegularType
from awkward.types.listtype import ListType
from awkward.types.optiontype import OptionType
from awkward.types.recordtype import RecordType
from awkward.types.uniontype import UnionType
from awkward.types.arraytype import ArrayType


grammar = r"""
start: type

type: numpytype
    | unknowntype
    | regulartype
    | listtype
    | varlen_string
    | fixedlen_string
    | char
    | byte
    | option1
    | option2
    | tuple
    | tuple_parameters
    | record
    | record_parameters
    | named0
    | named
    | union
    | list_parameters
    | categorical

numpytype: numpytype_name ("[" "parameters" "=" json_object "]")?

numpytype_name: DTYPE
              | DATETIME64
              | TIMEDELTA64

DTYPE.2: "bool"
       | "int8"
       | "uint8"
       | "int16"
       | "uint16"
       | "int32"
       | "uint32"
       | "int64"
       | "uint64"
       | "float32"
       | "float64"
       | "complex64"
       | "complex128"

DATETIME64:  /datetime64(\[(\s*-?[0-9]*)?(Y|M|W|D|h|m|s|ms|us|\u03bc|ns|ps|fs|as)\])?/
TIMEDELTA64: /timedelta64(\[(\s*-?[0-9]*)?(Y|M|W|D|h|m|s|ms|us|\u03bc|ns|ps|fs|as)\])?/

unknowntype: "unknown" ("[" "parameters" "=" json_object "]")?

regulartype: INT "*" type

listtype: "var" "*" type

varlen_string: "string" -> varlen_string
             | "bytes" -> varlen_bytestring

fixedlen_string: ("string" "[" INT "]") -> fixedlen_string
               | ("bytes" "[" INT "]") -> fixedlen_bytestring

char: "char"

byte: "byte"

option1: "?" type

option2: "option" "[" type ("," "parameters" "=" json_object)? "]"

tuple: "(" types? ")"
types: type ("," type)*

tuple_parameters: "tuple" "[" "[" types? "]" ("," "parameters" "=" json_object)? "]"

record: "{" pairs? "}"
pairs:  pair ("," pair)*
pair:   key ":" type
key:    ESCAPED_STRING -> string
      | CNAME          -> identifier

record_parameters: "struct" "[" "{" pairs? "}" ("," "parameters" "=" json_object)? "]"

named0:      CNAME "[" ("parameters" "=" json_object)? "]"
named:       CNAME "[" (named_types | named_pairs) "]"
named_types: type ("," (named_types | "parameters" "=" json_object))?
named_pairs: named_pair ("," (named_pairs | "parameters" "=" json_object))?
named_pair:  named_key ":" type
named_key:   ESCAPED_STRING -> string
           | CNAME          -> identifier

union: "union" "[" named_types? "]"

list_parameters: "[" type "," "parameters" "=" json_object "]"

categorical: "categorical" "[" "type" "=" type "]"

json: ESCAPED_STRING -> string
    | SIGNED_NUMBER  -> number
    | "true"         -> true
    | "false"        -> false
    | "null"         -> null
    | json_array
    | json_object

json_array:  "[" [json ("," json)*] "]"
json_object: "{" [json_pair ("," json_pair)*] "}"
json_pair:   ESCAPED_STRING ":" json

%import common.INT
%import common.CNAME
%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS

%ignore WS
"""


class Transformer:
    @staticmethod
    def _parameters(args, i):
        if i < len(args):
            return args[i]
        else:
            return None

    def start(self, args):
        return args[0]

    def type(self, args):
        return args[0]

    def numpytype(self, args):
        return ak.types.NumpyType(args[0], parameters=self._parameters(args, 1))

    def numpytype_name(self, args):
        return str(args[0])

    def unknowntype(self, args):
        return ak.types.UnknownType(parameters=self._parameters(args, 0))

    def regulartype(self, args):
        return ak.types.RegularType(args[1], int(args[0]))

    def listtype(self, args):
        return ak.types.ListType(args[0])

    def varlen_string(self, args):
        return ak.types.ListType(
            ak.types.NumpyType("uint8", {"__array__": "char"}),
            {"__array__": "string"},
        )

    def varlen_bytestring(self, args):
        return ak.types.ListType(
            ak.types.NumpyType("uint8", {"__array__": "byte"}),
            {"__array__": "bytestring"},
        )

    def fixedlen_string(self, args):
        return ak.types.RegularType(
            ak.types.NumpyType("uint8", {"__array__": "char"}),
            int(args[0]),
            {"__array__": "string"},
        )

    def fixedlen_bytestring(self, args):
        return ak.types.RegularType(
            ak.types.NumpyType("uint8", {"__array__": "byte"}),
            int(args[0]),
            {"__array__": "bytestring"},
        )

    def char(self, args):
        return ak.types.NumpyType("uint8", {"__array__": "char"})

    def byte(self, args):
        return ak.types.NumpyType("uint8", {"__array__": "byte"})

    def option1(self, args):
        return ak.types.OptionType(args[0])

    def option2(self, args):
        return ak.types.OptionType(args[0], parameters=self._parameters(args, 1))

    def tuple(self, args):
        if len(args) == 0:
            types = []
        else:
            types = args[0]
        return ak.types.RecordType(types, None)

    def types(self, args):
        return args

    def tuple_parameters(self, args):
        if len(args) != 0 and isinstance(args[0], list):
            types = args[0]
        else:
            types = []

        if len(args) != 0 and isinstance(args[-1], dict):
            parameters = args[-1]
        else:
            parameters = {}

        return ak.types.RecordType(types, None, parameters)

    def record(self, args):
        if len(args) == 0:
            fields = []
            types = []
        else:
            fields = [x[0] for x in args[0]]
            types = [x[1] for x in args[0]]
        return ak.types.RecordType(types, fields)

    def pairs(self, args):
        return args

    def pair(self, args):
        return tuple(args)

    def record_parameters(self, args):
        if len(args) != 0 and isinstance(args[0], list):
            fields = [x[0] for x in args[0]]
            types = [x[1] for x in args[0]]
        else:
            fields = []
            types = []

        if len(args) != 0 and isinstance(args[-1], dict):
            parameters = args[-1]
        else:
            parameters = {}

        return ak.types.RecordType(types, fields, parameters)

    def named0(self, args):
        parameters = {"__record__": str(args[0])}
        if 1 < len(args):
            parameters.update(args[1])
        return ak.types.RecordType([], None, parameters)

    def named(self, args):
        parameters = {"__record__": str(args[0])}

        if isinstance(args[1][-1], dict):
            arguments = args[1][:-1]
            parameters.update(args[1][-1])
        else:
            arguments = args[1]

        if any(isinstance(x, tuple) for x in arguments):
            fields = [x[0] for x in arguments]
            contents = [x[1] for x in arguments]
        else:
            fields = None
            contents = arguments

        return ak.types.RecordType(contents, fields, parameters)

    def named_types(self, args):
        if len(args) == 2 and isinstance(args[1], list):
            return args[:1] + args[1]
        else:
            return args

    def named_pairs(self, args):
        if len(args) == 2 and isinstance(args[1], list):
            return args[:1] + args[1]
        else:
            return args

    def named_pair(self, args):
        return tuple(args)

    def identifier(self, args):
        return str(args[0])

    def union(self, args):
        if len(args) == 0:
            arguments = []
            parameters = None
        elif isinstance(args[0][-1], dict):
            arguments = args[0][:-1]
            parameters = args[0][-1]
        else:
            arguments = args[0]
            parameters = None

        return ak.types.UnionType(arguments, parameters)

    def list_parameters(self, args):
        # modify recently created type object
        args[0].parameters.update(args[1])
        return args[0]

    def categorical(self, args):
        # modify recently created type object
        args[0].parameters["__categorical__"] = True
        return args[0]

    def json(self, args):
        return args[0]

    def json_object(self, args):
        return dict(args)

    def json_pair(self, args):
        return (json.loads(args[0]), args[1])

    def json_array(self, args):
        return list(args)

    def string(self, args):
        return json.loads(args[0])

    def number(self, args):
        try:
            return int(args[0])
        except ValueError:
            return float(args[0])

    def true(self, args):
        return True

    def false(self, args):
        return False

    def null(self, args):
        return None


parser = lark.Lark(grammar, parser="lalr", transformer=Transformer())


def test_numpytype_int32():
    t = NumpyType("int32")
    assert str(parser.parse(str(t))) == str(t)


def test_numpytype_datetime64():
    t = NumpyType("datetime64")
    assert str(parser.parse(str(t))) == str(t)


def test_numpytype_datetime64_10s():
    t = NumpyType("datetime64[10s]")
    assert str(parser.parse(str(t))) == str(t)


def test_numpytype_int32_parameter():
    t = NumpyType("int32", {"__array__": "Something"})
    assert str(parser.parse(str(t))) == str(t)


def test_numpytype_datetime64_parameter():
    t = NumpyType("datetime64", {"__array__": "Something"})
    assert str(parser.parse(str(t))) == str(t)


def test_numpytype_datetime64_10s_parameter():
    t = NumpyType("datetime64[10s]", {"__array__": "Something"})
    assert str(parser.parse(str(t))) == str(t)


def test_numpytype_int32_categorical():
    t = NumpyType("int32", {"__categorical__": True})
    assert str(parser.parse(str(t))) == str(t)


def test_numpytype_int32_parameters_categorical():
    t = NumpyType("int32", {"__array__": "Something", "__categorical__": True})
    assert str(parser.parse(str(t))) == str(t)


def test_unknowntype():
    t = UnknownType()
    assert str(parser.parse(str(t))) == str(t)


def test_unknowntype_parameter():
    t = UnknownType({"__array__": "Something"})
    assert str(parser.parse(str(t))) == str(t)


def test_unknowntype_categorical():
    t = UnknownType({"__categorical__": True})
    assert str(parser.parse(str(t))) == str(t)


def test_unknowntype_categorical_parameter():
    t = UnknownType({"__array__": "Something", "__categorical__": True})
    assert str(parser.parse(str(t))) == str(t)


def test_regulartype_numpytype():
    t = RegularType(NumpyType("int32"), 5)
    assert str(parser.parse(str(t))) == str(t)


def test_regulartype_numpytype_parameter():
    t = RegularType(NumpyType("int32"), 5, {"__array__": "Something"})
    assert str(parser.parse(str(t))) == str(t)


def test_regulartype_numpytype_categorical():
    t = RegularType(NumpyType("int32"), 5, {"__categorical__": True})
    assert str(parser.parse(str(t))) == str(t)


def test_regulartype_numpytype_categorical_parameter():
    t = RegularType(
        NumpyType("int32"), 5, {"__categorical__": True, "__array__": "Something"}
    )
    assert str(parser.parse(str(t))) == str(t)


def test_listtype_numpytype():
    t = ListType(NumpyType("int32"))
    assert str(parser.parse(str(t))) == str(t)


def test_listtype_numpytype_parameter():
    t = ListType(NumpyType("int32"), {"__array__": "Something"})
    assert str(parser.parse(str(t))) == str(t)


def test_listtype_numpytype_categorical():
    t = ListType(NumpyType("int32"), {"__categorical__": True})
    assert str(parser.parse(str(t))) == str(t)


def test_listtype_numpytype_categorical_parameter():
    t = ListType(
        NumpyType("int32"), {"__categorical__": True, "__array__": "Something"}
    )
    assert str(parser.parse(str(t))) == str(t)


def test_varlen_string():
    t = ListType(NumpyType("uint8", {"__array__": "char"}), {"__array__": "string"})
    assert str(parser.parse(str(t))) == str(t)


def test_varlen_bytestring():
    t = ListType(NumpyType("uint8", {"__array__": "char"}), {"__array__": "bytestring"})
    assert str(parser.parse(str(t))) == str(t)


def test_fixedlen_string():
    t = RegularType(
        NumpyType("uint8", {"__array__": "char"}), 5, {"__array__": "string"}
    )
    assert str(parser.parse(str(t))) == str(t)


def test_fixedlen_bytestring():
    t = RegularType(
        NumpyType("uint8", {"__array__": "byte"}), 5, {"__array__": "bytestring"}
    )
    assert str(parser.parse(str(t))) == str(t)


def test_char():
    t = NumpyType("uint8", {"__array__": "char"})
    assert str(parser.parse(str(t))) == str(t)


def test_byte():
    t = NumpyType("uint8", {"__array__": "byte"})
    assert str(parser.parse(str(t))) == str(t)


def test_optiontype_numpytype_int32():
    t = OptionType(NumpyType("int32"))
    assert str(parser.parse(str(t))) == str(t)


def test_optiontype_numpytype_int32_parameters():
    t = OptionType(NumpyType("int32"), {"__array__": "Something"})
    assert str(parser.parse(str(t))) == str(t)


def test_optiontype_numpytype_int32_categorical():
    t = OptionType(NumpyType("int32"), {"__categorical__": True})
    assert str(parser.parse(str(t))) == str(t)


def test_optiontype_numpytype_int32_categorical_parameters():
    t = OptionType(
        NumpyType("int32"), {"__array__": "Something", "__categorical__": True}
    )
    assert str(parser.parse(str(t))) == str(t)


def test_option_varlen_string():
    t = OptionType(
        ListType(NumpyType("uint8", {"__array__": "char"}), {"__array__": "string"})
    )
    assert str(parser.parse(str(t))) == str(t)


def test_option_varlen_string_parameters():
    t = OptionType(
        ListType(NumpyType("uint8", {"__array__": "char"}), {"__array__": "string"}),
        {"__array__": "Something"},
    )
    assert str(parser.parse(str(t))) == str(t)


def test_record_empty():
    t = RecordType([], None)
    assert str(parser.parse(str(t))) == str(t)


def test_record_fields_empty():
    t = RecordType([], [])
    assert str(parser.parse(str(t))) == str(t)


def test_record_int32():
    t = RecordType([NumpyType("int32")], None)
    assert str(parser.parse(str(t))) == str(t)


def test_record_int32_float64():
    t = RecordType([NumpyType("int32"), NumpyType("float64")], None)
    assert str(parser.parse(str(t))) == str(t)


def test_record_fields_int32():
    t = RecordType([NumpyType("int32")], ["one"])
    assert str(parser.parse(str(t))) == str(t)


def test_record_fields_int32_float64():
    t = RecordType([NumpyType("int32"), NumpyType("float64")], ["one", "t w o"])
    assert str(parser.parse(str(t))) == str(t)


def test_record_empty_parameters():
    t = RecordType([], None, {"p": [123]})
    assert str(parser.parse(str(t))) == str(t)


def test_record_fields_empty_parameters():
    t = RecordType([], [], {"p": [123]})
    assert str(parser.parse(str(t))) == str(t)


def test_record_int32_parameters():
    t = RecordType([NumpyType("int32")], None, {"p": [123]})
    assert str(parser.parse(str(t))) == str(t)


def test_record_int32_float64_parameters():
    t = RecordType([NumpyType("int32"), NumpyType("float64")], None, {"p": [123]})
    assert str(parser.parse(str(t))) == str(t)


def test_record_fields_int32_parameters():
    t = RecordType([NumpyType("int32")], ["one"], {"p": [123]})
    assert str(parser.parse(str(t))) == str(t)


def test_record_fields_int32_float64_parameters():
    t = RecordType(
        [NumpyType("int32"), NumpyType("float64")], ["one", "t w o"], {"p": [123]}
    )
    assert str(parser.parse(str(t))) == str(t)


def test_named_record_empty():
    t = RecordType([], None, {"__record__": "Name"})
    assert str(parser.parse(str(t))) == str(t)


def test_named_record_int32():
    t = RecordType([NumpyType("int32")], None, {"__record__": "Name"})
    assert str(parser.parse(str(t))) == str(t)


def test_named_record_int32_float64():
    t = RecordType(
        [NumpyType("int32"), NumpyType("float64")], None, {"__record__": "Name"}
    )
    assert str(parser.parse(str(t))) == str(t)


def test_named_record_fields_int32():
    t = RecordType([NumpyType("int32")], ["one"], {"__record__": "Name"})
    assert str(parser.parse(str(t))) == str(t)


def test_named_record_fields_int32_float64():
    t = RecordType(
        [NumpyType("int32"), NumpyType("float64")],
        ["one", "t w o"],
        {"__record__": "Name"},
    )
    assert str(parser.parse(str(t))) == str(t)


def test_named_record_empty_parameters():
    t = RecordType([], None, {"__record__": "Name", "p": [123]})
    assert str(parser.parse(str(t))) == str(t)


def test_named_record_int32_parameters():
    t = RecordType([NumpyType("int32")], None, {"__record__": "Name", "p": [123]})
    assert str(parser.parse(str(t))) == str(t)


def test_named_record_int32_float64_parameters():
    t = RecordType(
        [NumpyType("int32"), NumpyType("float64")],
        None,
        {"__record__": "Name", "p": [123]},
    )
    assert str(parser.parse(str(t))) == str(t)


def test_named_record_fields_int32_parameters():
    t = RecordType([NumpyType("int32")], ["one"], {"__record__": "Name", "p": [123]})
    assert str(parser.parse(str(t))) == str(t)


def test_named_record_fields_int32_float64_parameters():
    t = RecordType(
        [NumpyType("int32"), NumpyType("float64")],
        ["one", "t w o"],
        {"__record__": "Name", "p": [123]},
    )
    assert str(parser.parse(str(t))) == str(t)


def test_union_empty():
    t = UnionType([])
    assert str(parser.parse(str(t))) == str(t)


def test_union_float64():
    t = UnionType([NumpyType("float64")])
    assert str(parser.parse(str(t))) == str(t)


def test_union_float64_datetime64():
    t = UnionType(
        [NumpyType("float64"), NumpyType("datetime64")],
    )
    assert str(parser.parse(str(t))) == str(t)


def test_union_float64_parameters():
    t = UnionType([NumpyType("float64")], {"__array__": "Something"})
    assert str(parser.parse(str(t))) == str(t)


def test_union_float64_datetime64_parameters():
    t = UnionType(
        [NumpyType("float64"), NumpyType("datetime64")],
        {"__array__": "Something"},
    )
    assert str(parser.parse(str(t))) == str(t)
