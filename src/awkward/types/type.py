# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import json
import sys

import awkward as ak
from awkward.types._awkward_datashape_parser import Lark_StandAlone, Transformer

np = ak._nplikes.NumpyMetadata.instance()


class Type:
    @property
    def parameters(self):
        if self._parameters is None:  # pylint: disable=E0203
            self._parameters = {}
        return self._parameters

    def parameter(self, key):
        if self._parameters is None:
            return None
        else:
            return self._parameters.get(key)

    @property
    def typestr(self):
        return self._typestr

    def __str__(self):
        return "".join(self._str("", True))

    def show(self, stream=sys.stdout):
        stream.write("".join(self._str("", False) + ["\n"]))

    _str_parameters_exclude = ("__categorical__",)

    def _str_categorical_begin(self):
        if self.parameter("__categorical__") is not None:
            return "categorical[type="
        else:
            return ""

    def _str_categorical_end(self):
        if self.parameter("__categorical__") is not None:
            return "]"
        else:
            return ""

    def _str_parameters(self):
        out = []
        if self._parameters is not None:
            for k, v in self._parameters.items():
                if k not in self._str_parameters_exclude:
                    out.append(json.dumps(k) + ": " + json.dumps(v))

        if len(out) == 0:
            return None
        else:
            return "parameters={" + ", ".join(out) + "}"

    def _repr_args(self):
        out = []

        if self._parameters is not None and len(self._parameters) > 0:
            out.append("parameters=" + repr(self._parameters))

        if self._typestr is not None:
            out.append("typestr=" + repr(self._typestr))

        return out


class _DataShapeTransformer(Transformer):
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
            ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
            parameters={"__array__": "string"},
        )

    def varlen_bytestring(self, args):
        return ak.types.ListType(
            ak.types.NumpyType("uint8", parameters={"__array__": "byte"}),
            parameters={"__array__": "bytestring"},
        )

    def fixedlen_string(self, args):
        return ak.types.RegularType(
            ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
            int(args[0]),
            parameters={"__array__": "string"},
        )

    def fixedlen_bytestring(self, args):
        return ak.types.RegularType(
            ak.types.NumpyType("uint8", parameters={"__array__": "byte"}),
            int(args[0]),
            parameters={"__array__": "bytestring"},
        )

    def char(self, args):
        return ak.types.NumpyType("uint8", parameters={"__array__": "char"})

    def byte(self, args):
        return ak.types.NumpyType("uint8", parameters={"__array__": "byte"})

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

        return ak.types.RecordType(types, None, parameters=parameters)

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

        return ak.types.RecordType(types, fields, parameters=parameters)

    def named0(self, args):
        parameters = {"__record__": str(args[0])}
        if 1 < len(args):
            parameters.update(args[1])
        return ak.types.RecordType([], None, parameters=parameters)

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

        return ak.types.RecordType(contents, fields, parameters=parameters)

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

        return ak.types.UnionType(arguments, parameters=parameters)

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


def from_datashape(datashape, highlevel=True):
    """
    Parses `datashape` (str) and returns a #ak.types.Type object, the inverse of
    calling `str` on a #ak.types.Type.

    If `highlevel=True`, and the type string starts with a number (e.g. '1000 * ...'),
    the return type is #ak.types.ArrayType, representing an #ak.highlevel.Array.

    If `highlevel=True` and the type string starts with a record indicator (e.g. `{`),
    the return type is #ak.types.RecordType, representing an #ak.highlevel.Record,
    rather than an array of them.

    Other strings (e.g. starting with `var *`, `?`, `option`, etc.) are not compatible
    with `highlevel=True`; an exception would be raised.

    If `highlevel=False`, the type is assumed to represent a layout (e.g. a number
    indicates a #ak.types.RegularType, rather than a #ak.types.ArrayType).
    """
    from awkward.types.arraytype import ArrayType
    from awkward.types.recordtype import RecordType
    from awkward.types.regulartype import RegularType

    parser = Lark_StandAlone(transformer=_DataShapeTransformer())
    out = parser.parse(datashape)

    if highlevel:
        if isinstance(out, RegularType):
            return ArrayType(out.content, out.size)
        elif isinstance(out, RecordType):
            return out
        else:
            raise ak._errors.wrap_error(
                ValueError(
                    f"type '{type(out).__name__}' is not compatible with highlevel=True"
                )
            )

    else:
        return out
