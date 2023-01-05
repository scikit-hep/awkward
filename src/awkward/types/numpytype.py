# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import json

import awkward as ak
from awkward._nplikes import metadata
from awkward.forms.form import _parameters_equal
from awkward.types.type import Type
from awkward.typing import final

primitive_names = {
    d.name for d in metadata.all_dtypes if not metadata.isdtype(d, "timelike")
}


def is_primitive(primitive):
    if primitive in primitive_names:
        return True

    # TODO: is this maybe-timelike path too slow for from_buffers?
    try:
        dtype = metadata.dtype(primitive)
    except TypeError:
        return False

    return metadata.isdtype(dtype, "timelike")


def primitive_to_dtype(primitive):
    try:
        dtype = metadata.dtype(primitive)
    except TypeError as err:
        raise ak._errors.wrap_error(err) from None
    else:
        if metadata.isdtype(dtype, ("numeric", "bool", "timelike")):
            return dtype
        else:
            raise ak._errors.wrap_error(
                TypeError(
                    "unrecognized primitive: {}. Must be one of\n\n    {}\n\nor a "
                    "datetime64/timedelta64 with units (e.g. 'datetime64[15us]')".format(
                        repr(primitive), ", ".join(primitive_names)
                    )
                )
            )


def dtype_to_primitive(dtype):
    primitive = dtype.name

    if metadata.isdtype(dtype, ("numeric", "bool", "timelike")):
        return primitive
    else:
        raise ak._errors.wrap_error(
            TypeError(
                "unrecognized dtype: {}. Must be one of\n\n    {}\n\nor a "
                "datetime64/timedelta64 with units (e.g. 'datetime64[15us]')".format(
                    repr(primitive), ", ".join(primitive_names)
                )
            )
        )


@final
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
                numpy_unit = str(
                    metadata.dtype("M8[" + self._parameters["__unit__"] + "]")
                )
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
