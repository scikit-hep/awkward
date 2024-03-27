# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from awkward._behavior import find_array_typestr
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._parameters import parameters_are_equal, type_parameters_equal
from awkward._typing import Any, JSONMapping, cast, final
from awkward._util import UNSET, Sentinel
from awkward.types.type import Type

np = NumpyMetadata.instance()


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
            raise TypeError(
                "unrecognized primitive: {}. Must be one of\n\n    {}\n\nor a "
                "datetime64/timedelta64 with units (e.g. 'datetime64[15us]')".format(
                    repr(primitive), ", ".join(_primitive_to_dtype_dict)
                )
            )
        return out


def dtype_to_primitive(dtype):
    if dtype.kind.upper() == "M" and dtype == dtype.newbyteorder("="):
        return str(dtype)
    else:
        out = _dtype_to_primitive_dict.get(dtype)
        if out is None:
            raise TypeError(
                "unsupported dtype: {}. Must be one of\n\n    {}\n\nor a "
                "datetime64/timedelta64 with units (e.g. 'datetime64[15us]')".format(
                    repr(dtype), ", ".join(_primitive_to_dtype_dict)
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


@final
class NumpyType(Type):
    def copy(
        self,
        *,
        primitive: str | Sentinel = UNSET,
        parameters: JSONMapping | Sentinel | None = UNSET,
    ) -> NumpyType:
        return NumpyType(
            self._primitive if primitive is UNSET else primitive,  # type: ignore[arg-type]
            parameters=self._parameters if parameters is UNSET else parameters,  # type: ignore[arg-type]
        )

    def __init__(self, primitive: str, *, parameters: JSONMapping | None = None):
        primitive = dtype_to_primitive(primitive_to_dtype(primitive))
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError(
                f"{type(self).__name__} 'parameters' must be of type Mapping or None, not {parameters!r}"
            )
        self._primitive: str = primitive
        self._parameters: JSONMapping | None = parameters

    @property
    def primitive(self) -> str:
        return self._primitive

    _str_parameters_exclude: tuple[str, ...] = ("__categorical__", "__unit__")

    def _get_typestr(self, behavior: Mapping | None) -> str | None:
        typestr = find_array_typestr(behavior, self._parameters)
        if typestr is not None:
            return typestr

        if self._parameters is None:
            return None

        name = cast("str | None", self._parameters.get("__array__"))
        if name in {"byte", "char"}:
            return name

        return None

    def _str(self, indent: str, compact: bool, behavior: Mapping | None) -> list[str]:
        typestr = self._get_typestr(behavior)
        if typestr is not None:
            out = [typestr]

        else:
            if (unit := cast(str, self.parameter("__unit__"))) is not None:
                numpy_unit = str(np.dtype(f"M8[{unit}]"))
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
                out = [self._primitive, "[", units, params, "]"]  # type: ignore[list-item]

        return [self._str_categorical_begin(), *out, self._str_categorical_end()]

    def __repr__(self) -> str:
        args = [repr(self._primitive), *self._repr_args()]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _is_equal_to(self, other: Any, all_parameters: bool) -> bool:
        compare_parameters = (
            parameters_are_equal if all_parameters else type_parameters_equal
        )
        return (
            isinstance(other, type(self))
            and (self._primitive == other._primitive)
            and compare_parameters(self._parameters, other._parameters)
        )
