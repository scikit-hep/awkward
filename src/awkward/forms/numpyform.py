# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import awkward as ak
from awkward._meta.numpymeta import NumpyMeta
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.shape import unknown_length
from awkward._typing import TYPE_CHECKING, Any, DType, JSONMapping, Self, final
from awkward._util import UNSET
from awkward.forms.form import Form, _SpecifierMatcher

__all__ = ("NumpyForm",)

if TYPE_CHECKING:
    from awkward.forms.regularform import RegularForm

np = NumpyMetadata.instance()


def from_dtype(
    dtype,
    parameters: JSONMapping | None = None,
    *,
    time_units_as_parameter: bool = False,
):
    if dtype.subdtype is None:
        inner_shape = ()
    else:
        inner_shape = dtype.shape
        dtype = dtype.subdtype[0]

    if time_units_as_parameter:
        raise ValueError(
            "`time_units_as_parameter=True` is no longer supported; NumPy's time units are no longer converted into Awkward parameters"
        )

    return NumpyForm(
        primitive=ak.types.numpytype.dtype_to_primitive(dtype),
        parameters=parameters,
        inner_shape=inner_shape,
    )


@final
class NumpyForm(NumpyMeta, Form):
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
                f"{type(self).__name__} 'inner_shape' must be iterable, not {inner_shape!r}"
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
                out["inner_shape"] = [
                    None if item is unknown_length else item
                    for item in self._inner_shape
                ]
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

    def to_RegularForm(self) -> RegularForm | NumpyForm:
        out: RegularForm | NumpyForm = NumpyForm(
            self._primitive, (), parameters=None, form_key=None
        )
        for x in self._inner_shape[::-1]:
            out = ak.forms.RegularForm(out, x, parameters=None, form_key=None)
        out._parameters = self._parameters
        return out

    def _columns(self, path, output, list_indicator):
        output.append(".".join(path))

    def _select_columns(self, match_specifier: _SpecifierMatcher) -> Self:
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
        self, getkey: Callable[[Form, str], str], recursive: bool
    ) -> Iterator[tuple[str, DType]]:
        from awkward.types.numpytype import primitive_to_dtype

        yield (getkey(self, "data"), primitive_to_dtype(self.primitive))

    def _is_equal_to(self, other: Any, all_parameters: bool, form_key: bool) -> bool:
        return self._is_equal_to_generic(other, all_parameters, form_key) and (
            self._primitive == other._primitive
        )
