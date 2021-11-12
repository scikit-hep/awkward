# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from awkward._v2.contents.content import NestedIndexError
from awkward._v2.forms.form import Form, _parameters_equal

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


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
        primitive=ak._v2.types.numpytype.dtype_to_primitive(dtype),
        parameters=parameters,
        inner_shape=inner_shape,
    )


class NumpyForm(Form):
    def __init__(
        self,
        primitive,
        inner_shape=(),
        has_identifier=False,
        parameters=None,
        form_key=None,
    ):
        primitive = ak._v2.types.numpytype.dtype_to_primitive(
            ak._v2.types.numpytype.primitive_to_dtype(primitive)
        )
        if not isinstance(inner_shape, Iterable):
            raise TypeError(
                "{0} 'inner_shape' must be iterable, not {1}".format(
                    type(self).__name__, repr(inner_shape)
                )
            )

        self._primitive = primitive
        self._inner_shape = tuple(inner_shape)
        self._init(has_identifier, parameters, form_key)

    @property
    def primitive(self):
        return self._primitive

    @property
    def inner_shape(self):
        return self._inner_shape

    @property
    def itemsize(self):
        return ak._v2.types.numpytype.primitive_to_dtype(self._primitive).itemsize

    def __repr__(self):
        args = [repr(self._primitive)]
        if len(self._inner_shape) > 0:
            args.append("inner_shape=" + repr(self._inner_shape))
        args += self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def _tolist_part(self, verbose, toplevel):
        if (
            not verbose
            and not toplevel
            and len(self._inner_shape) == 0
            and not self._has_identifier
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
            return self._tolist_extra(out, verbose)

    def _type(self, typestrs):
        out = ak._v2.types.numpytype.NumpyType(
            self._primitive,
            None,
            ak._v2._util.gettypestr(self._parameters, typestrs),
        )
        for x in self._inner_shape[::-1]:
            out = ak._v2.types.regulartype.RegularType(out, x)
        out._parameters = self._parameters
        return out

    def __eq__(self, other):
        if isinstance(other, NumpyForm):
            return (
                self._has_identifier == other._has_identifier
                and self._form_key == other._form_key
                and self._primitive == other._primitive
                and self._inner_shape == other._inner_shape
                and _parameters_equal(self._parameters, other._parameters)
            )
        else:
            return False

    def generated_compatibility(self, other):
        if other is None:
            return True

        elif isinstance(other, NumpyForm):
            return (
                ak._v2.types.numpytype.primitive_to_dtype(self._primitive)
                == ak._v2.types.numpytype.primitive_to_dtype(other._primitive)
                and self._inner_shape == other._inner_shape
                and _parameters_equal(self._parameters, other._parameters)
            )

        else:
            return False

    def _getitem_range(self):
        return NumpyForm(
            self._primitive,
            inner_shape=self._inner_shape,
            has_identifier=self._has_identifier,
            parameters=self._parameters,
            form_key=None,
        )

    def _getitem_field(self, where, only_fields=()):
        raise NestedIndexError(self, where, "not an array of records")

    def _getitem_fields(self, where, only_fields=()):
        raise NestedIndexError(self, where, "not an array of records")

    def _carry(self, allow_lazy):
        return NumpyForm(
            self._primitive,
            inner_shape=self._inner_shape,
            has_identifier=self._has_identifier,
            parameters=self._parameters,
            form_key=None,
        )

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
    def dimension_optiontype(self):
        return False
