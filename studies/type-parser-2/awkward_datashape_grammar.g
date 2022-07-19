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
