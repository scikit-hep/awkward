# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

__all__ = ("NumpyForm",)
from collections.abc import Callable, Iterable, Iterator

import awkward as ak
from awkward._errors import deprecate
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._parameters import type_parameters_equal
from awkward._typing import JSONSerializable, Self, final
from awkward._util import UNSET
from awkward.forms.form import Form

np = NumpyMetadata.instance()


def from_dtype(dtype, parameters=None, *, time_units_as_parameter: bool = UNSET):
    if dtype.subdtype is None:
        inner_shape = ()
    else:
        inner_shape = dtype.shape
        dtype = dtype.subdtype[0]

    if time_units_as_parameter is UNSET:
        time_units_as_parameter = True

    if time_units_as_parameter:
        deprecate(
            "from_dtype conversion of temporal units to generic `datetime64` and `timedelta64` types is deprecated, "
            "pass `time_units_as_parameter=False` to disable this warning.",
            version="2.4.0",
        )

    if time_units_as_parameter and issubclass(
        dtype.type, (np.datetime64, np.timedelta64)
    ):
        unit, step = np.datetime_data(dtype)
        if unit != "generic":
            unitstr = ("" if step == 1 else str(step)) + unit
            if parameters is None:
                parameters = {}
            else:
                parameters = parameters.copy()
            parameters["__unit__"] = unitstr
            dtype = np.dtype(dtype.type)

    return NumpyForm(
        primitive=ak.types.numpytype.dtype_to_primitive(dtype),
        parameters=parameters,
        inner_shape=inner_shape,
    )


@final
class NumpyForm(Form):
    is_numpy = True

    def __init__(
        self,
        primitive,
        inner_shape=(),
        *,
        parameters=None,
        form_key=None,
    ):
        primitive = ak.types.numpytype.dtype_to_primitive(
            ak.types.numpytype.primitive_to_dtype(primitive)
        )
        if not isinstance(inner_shape, Iterable):
            raise TypeError(
                "{} 'inner_shape' must be iterable, not {}".format(
                    type(self).__name__, repr(inner_shape)
                )
            )

        self._primitive = primitive
        self._inner_shape = tuple(inner_shape)
        self._init(parameters=parameters, form_key=form_key)

    @property
    def primitive(self):
        return self._primitive

    @property
    def inner_shape(self):
        return self._inner_shape

    def copy(
        self,
        primitive=UNSET,
        inner_shape=UNSET,
        *,
        parameters=UNSET,
        form_key=UNSET,
    ):
        return NumpyForm(
            self._primitive if primitive is UNSET else primitive,
            self._inner_shape if inner_shape is UNSET else inner_shape,
            parameters=self._parameters if parameters is UNSET else parameters,
            form_key=self._form_key if form_key is UNSET else form_key,
        )

    @classmethod
    def simplified(
        cls,
        primitive,
        inner_shape=(),
        *,
        parameters=None,
        form_key=None,
    ):
        return cls(primitive, inner_shape, parameters=parameters, form_key=form_key)

    @property
    def itemsize(self):
        return ak.types.numpytype.primitive_to_dtype(self._primitive).itemsize

    def __repr__(self):
        args = [repr(self._primitive)]
        if len(self._inner_shape) > 0:
            args.append("inner_shape=" + repr(self._inner_shape))
        args += self._repr_args()
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _to_dict_part(self, verbose, toplevel):
        if (
            not verbose
            and not toplevel
            and len(self._inner_shape) == 0
            and (self._parameters is None or len(self._parameters) == 0)
            and self._form_key is None
        ):
            return self._primitive

        else:
            out = {
                "class": "NumpyArray",
                "primitive": self._primitive,
            }
            if verbose or len(self._inner_shape) > 0:
                out["inner_shape"] = list(self._inner_shape)
            return self._to_dict_extra(out, verbose)

    @property
    def type(self):
        out = ak.types.NumpyType(
            self._primitive,
            parameters=None,
        )
        for x in self._inner_shape[::-1]:
            out = ak.types.RegularType(out, x)

        out._parameters = self._parameters

        return out

    def __eq__(self, other):
        if isinstance(other, NumpyForm):
            return (
                self._form_key == other._form_key
                and self._primitive == other._primitive
                and self._inner_shape == other._inner_shape
                and type_parameters_equal(self._parameters, other._parameters)
            )
        else:
            return False

    def to_RegularForm(self):
        out = NumpyForm(self._primitive, (), parameters=None, form_key=None)
        for x in self._inner_shape[::-1]:
            out = ak.forms.RegularForm(out, x, parameters=None, form_key=None)
        out._parameters = self._parameters
        return out

    def purelist_parameters(self, *keys: str) -> JSONSerializable:
        if self._parameters is not None:
            for key in keys:
                if key in self._parameters:
                    return self._parameters[key]

    @property
    def purelist_isregular(self):
        return True

    @property
    def purelist_depth(self):
        return len(self.inner_shape) + 1

    @property
    def is_identity_like(self):
        return False

    @property
    def minmax_depth(self):
        depth = len(self.inner_shape) + 1
        return (depth, depth)

    @property
    def branch_depth(self):
        return (False, len(self.inner_shape) + 1)

    @property
    def fields(self):
        return []

    @property
    def is_tuple(self):
        return False

    @property
    def dimension_optiontype(self):
        return False

    def _columns(self, path, output, list_indicator):
        output.append(".".join(path))

    def _select_columns(self, match_specifier):
        return self

    def _prune_columns(self, is_inside_record_or_union: bool) -> Self:
        return self

    def _column_types(self):
        return (ak.types.numpytype.primitive_to_dtype(self._primitive),)

    def __setstate__(self, state):
        if isinstance(state, dict):
            # read data pickled in Awkward 2.x
            self.__dict__.update(state)
        else:
            # read data pickled in Awkward 1.x

            # https://github.com/scikit-hep/awkward/blob/main-v1/src/python/forms.cpp#L530-L537
            has_identities, parameters, form_key, inner_shape, itemsize, format = state

            # https://github.com/scikit-hep/awkward/blob/main-v1/src/libawkward/util.cpp#L131-L145
            format = format.lstrip("<").lstrip(">").lstrip("=")

            # https://github.com/scikit-hep/awkward/blob/main-v1/src/libawkward/util.cpp#L147-L222
            if format == "?":
                dtype = np.dtype(np.bool_)
            elif format in ("b", "h", "i", "l", "q"):
                if itemsize == 1:
                    dtype = np.dtype(np.int8)
                elif itemsize == 2:
                    dtype = np.dtype(np.int16)
                elif itemsize == 4:
                    dtype = np.dtype(np.int32)
                elif itemsize == 8:
                    dtype = np.dtype(np.int64)
                else:
                    raise AssertionError(format)
            elif format in ("c", "B", "H", "I", "L", "Q"):
                if itemsize == 1:
                    dtype = np.dtype(np.uint8)
                elif itemsize == 2:
                    dtype = np.dtype(np.uint16)
                elif itemsize == 4:
                    dtype = np.dtype(np.uint32)
                elif itemsize == 8:
                    dtype = np.dtype(np.uint64)
                else:
                    raise AssertionError(format)
            elif format == "e":
                dtype = np.dtype(np.float16)
            elif format == "f":
                dtype = np.dtype(np.float32)
            elif format == "d":
                dtype = np.dtype(np.float64)
            elif format == "g":
                dtype = np.dtype(np.float128)
            elif format == "Zf":
                dtype = np.dtype(np.complex64)
            elif format == "Zd":
                dtype = np.dtype(np.complex128)
            elif format == "Zg":
                dtype = np.dtype(np.complex256)
            else:
                dtype = np.dtype(format)  # datetime or timedelta with units

            primitive = ak.types.numpytype.dtype_to_primitive(dtype)

            if form_key is not None:
                form_key = "part0-" + form_key  # only the first partition

            self.__init__(
                primitive, inner_shape, parameters=parameters, form_key=form_key
            )

    def _expected_from_buffers(
        self, getkey: Callable[[Form, str], str]
    ) -> Iterator[tuple[str, np.dtype]]:
        from awkward.types.numpytype import primitive_to_dtype

        yield (getkey(self, "data"), primitive_to_dtype(self.primitive))
