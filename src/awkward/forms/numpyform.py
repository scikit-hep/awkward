# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from collections.abc import Iterable

import awkward as ak
from awkward._util import unset
from awkward.forms.form import Form, _parameters_equal

np = ak._nplikes.NumpyMetadata.instance()


def from_dtype(dtype, parameters=None):
    if dtype.subdtype is None:
        inner_shape = ()
    else:
        inner_shape = dtype.shape
        dtype = dtype.subdtype[0]

    if issubclass(dtype.type, (np.datetime64, np.timedelta64)):
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
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'inner_shape' must be iterable, not {}".format(
                        type(self).__name__, repr(inner_shape)
                    )
                )
            )

        self._primitive = primitive
        self._inner_shape = tuple(inner_shape)
        self._init(parameters, form_key)

    @property
    def primitive(self):
        return self._primitive

    @property
    def inner_shape(self):
        return self._inner_shape

    def copy(
        self,
        primitive=unset,
        inner_shape=unset,
        *,
        parameters=unset,
        form_key=unset,
    ):
        return NumpyForm(
            self._primitive if primitive is unset else primitive,
            self._inner_shape if inner_shape is unset else inner_shape,
            parameters=self._parameters if parameters is unset else parameters,
            form_key=self._form_key if form_key is unset else form_key,
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

    def _type(self, typestrs):
        out = ak.types.NumpyType(
            self._primitive,
            parameters=None,
            typestr=ak._util.gettypestr(self._parameters, typestrs),
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
                and _parameters_equal(
                    self._parameters, other._parameters, only_array_record=True
                )
            )
        else:
            return False

    def to_RegularForm(self):
        out = NumpyForm(self._primitive, (), parameters=None, form_key=None)
        for x in self._inner_shape[::-1]:
            out = ak.forms.RegularForm(out, x, parameters=None, form_key=None)
        out._parameters = self._parameters
        return out

    def purelist_parameter(self, key):
        if self._parameters is None or key not in self._parameters:
            return None
        else:
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

    def _select_columns(self, index, specifier, matches, output):
        if any(match and index >= len(item) for item, match in zip(specifier, matches)):
            output.append(None)
        return self

    def _column_types(self):
        return (ak.types.numpytype.primitive_to_dtype(self._primitive),)
