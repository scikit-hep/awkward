# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import json
import re

import awkward as ak
from awkward.forms.form import _parameters_equal
from awkward.types.type import Type

np = ak._nplikes.NumpyMetadata.instance()


def is_primitive(primitive):
    if _primitive_to_dtype_datetime.match(primitive) is not None:
        return True
    elif _primitive_to_dtype_timedelta.match(primitive) is not None:
        return True
    else:
        return primitive in _primitive_to_dtype_dict


def primitive_to_dtype(primitive):
    if _primitive_to_dtype_datetime.match(primitive) is not None:
        return np.dtype(primitive)
    elif _primitive_to_dtype_timedelta.match(primitive) is not None:
        return np.dtype(primitive)
    else:
        out = _primitive_to_dtype_dict.get(primitive)
        if out is None:
            raise ak._errors.wrap_error(
                TypeError(
                    "unrecognized primitive: {}. Must be one of\n\n    {}\n\nor a "
                    "datetime64/timedelta64 with units (e.g. 'datetime64[15us]')".format(
                        repr(primitive), ", ".join(_primitive_to_dtype_dict)
                    )
                )
            )
        return out


def dtype_to_primitive(dtype):
    if dtype.kind.upper() == "M" and dtype == dtype.newbyteorder("="):
        return str(dtype)
    else:
        out = _dtype_to_primitive_dict.get(dtype)
        if out is None:
            raise ak._errors.wrap_error(
                TypeError(
                    "unsupported dtype: {}. Must be one of\n\n    {}\n\nor a "
                    "datetime64/timedelta64 with units (e.g. 'datetime64[15us]')".format(
                        repr(dtype), ", ".join(_primitive_to_dtype_dict)
                    )
                )
            )
        return out


_primitive_to_dtype_datetime = re.compile(
    r"datetime64\[(\s*-?[0-9]*)?(Y|M|W|D|h|m|s|ms|us|\u03bc|ns|ps|fs|as)\]"
)
_primitive_to_dtype_timedelta = re.compile(
    r"timedelta64\[(\s*-?[0-9]*)?(Y|M|W|D|h|m|s|ms|us|\u03bc|ns|ps|fs|as)\]"
)

_primitive_to_dtype_dict = {
    "bool": np.dtype(np.bool_),
    "int8": np.dtype(np.int8),
    "uint8": np.dtype(np.uint8),
    "int16": np.dtype(np.int16),
    "uint16": np.dtype(np.uint16),
    "int32": np.dtype(np.int32),
    "uint32": np.dtype(np.uint32),
    "int64": np.dtype(np.int64),
    "uint64": np.dtype(np.uint64),
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
    "complex64": np.dtype(np.complex64),
    "complex128": np.dtype(np.complex128),
    "datetime64": np.dtype(np.datetime64),
    "timedelta64": np.dtype(np.timedelta64),
}

if hasattr(np, "float16"):
    _primitive_to_dtype_dict["float16"] = np.dtype(np.float16)
if hasattr(np, "float128"):
    _primitive_to_dtype_dict["float128"] = np.dtype(np.float128)
if hasattr(np, "complex256"):
    _primitive_to_dtype_dict["complex256"] = np.dtype(np.complex256)

_dtype_to_primitive_dict = {}
for primitive, dtype in _primitive_to_dtype_dict.items():
    _dtype_to_primitive_dict[dtype] = primitive


class NumpyType(Type):
    def __init__(self, primitive, *, parameters=None, typestr=None):
        primitive = dtype_to_primitive(primitive_to_dtype(primitive))
        if parameters is not None and not isinstance(parameters, dict):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'parameters' must be of type dict or None, not {}".format(
                        type(self).__name__, repr(parameters)
                    )
                )
            )
        if typestr is not None and not isinstance(typestr, str):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'typestr' must be of type string or None, not {}".format(
                        type(self).__name__, repr(typestr)
                    )
                )
            )
        self._primitive = primitive
        self._parameters = parameters
        self._typestr = typestr

    @property
    def primitive(self):
        return self._primitive

    _str_parameters_exclude = ("__categorical__", "__unit__")

    def _str(self, indent, compact):
        if self._typestr is not None:
            out = [self._typestr]

        elif self.parameter("__array__") == "char":
            out = ["char"]

        elif self.parameter("__array__") == "byte":
            out = ["byte"]

        else:
            if self.parameter("__unit__") is not None:
                numpy_unit = str(np.dtype("M8[" + self._parameters["__unit__"] + "]"))
                bracket_index = numpy_unit.index("[")
                units = "unit=" + json.dumps(numpy_unit[bracket_index + 1 : -1])
            else:
                units = None

            params = self._str_parameters()

            if units is None and params is None:
                out = [self._primitive]
            else:
                if units is not None and params is not None:
                    units = units + ", "
                elif units is None:
                    units = ""
                elif params is None:
                    params = ""
                out = [self._primitive, "[", units, params, "]"]

        return [self._str_categorical_begin()] + out + [self._str_categorical_end()]

    def __repr__(self):
        args = [repr(self._primitive)] + self._repr_args()
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def __eq__(self, other):
        if isinstance(other, NumpyType):
            return self._primitive == other._primitive and _parameters_equal(
                self._parameters, other._parameters, only_array_record=True
            )
        else:
            return False
