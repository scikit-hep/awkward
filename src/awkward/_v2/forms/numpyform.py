# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from awkward._v2.forms.form import Form

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def from_dtype(dtype, parameters=None, inner_shape=None):
    if dtype.subdtype is not None:
        inner_shape = dtype.shape
        dtype = dtype.subdtype[0]
    if str(dtype) in ["datetime64[s]", "timedelta64[s]"] and "[" in str(dtype):
        dtype = str(dtype).split("[")
        parameters = parameters if parameters is not None else {}
        parameters["__unit__"] = dtype[1][:-1]
        dtype = dtype[0]
        return NumpyForm(
            primitive=dtype, parameters=parameters, inner_shape=inner_shape
        )
    return NumpyForm(
        primitive=ak._v2.types.numpytype._dtype_to_primitive[dtype],
        parameters=parameters,
        inner_shape=inner_shape,
    )


class NumpyForm(Form):
    def __init__(
        self,
        primitive,
        inner_shape=None,
        has_identities=False,
        parameters=None,
        form_key=None,
    ):
        if primitive not in ak._v2.types.numpytype._primitive_to_dtype:
            raise TypeError(
                "{0} 'primitive' must be one of {1}, not {2}".format(
                    type(self).__name__,
                    ", ".join(
                        repr(x) for x in ak._v2.types.numpytype._primitive_to_dtype
                    ),
                    repr(primitive),
                )
            )
        if inner_shape is not None and not isinstance(inner_shape, Iterable):
            raise TypeError(
                "{0} 'inner_shape' must be iterable, not {1}".format(
                    type(self).__name__, repr(inner_shape)
                )
            )
        if inner_shape is not None and not isinstance(inner_shape, list):
            inner_shape = list(inner_shape)
        if has_identities is not None and not isinstance(has_identities, bool):
            raise TypeError(
                "{0} 'has_identities' must be of type bool or None, not {1}".format(
                    type(self).__name__, repr(has_identities)
                )
            )
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError(
                "{0} 'parameters' must be of type dict or None, not {1}".format(
                    type(self).__name__, repr(parameters)
                )
            )
        if form_key is not None and not isinstance(form_key, str):
            raise TypeError(
                "{0} 'form_key' must be of type string or None, not {1}".format(
                    type(self).__name__, repr(form_key)
                )
            )
        self._primitive = primitive
        self._inner_shape = inner_shape
        self._has_identities = has_identities
        self._parameters = parameters
        self._form_key = form_key

    @property
    def primitive(self):
        return self._primitive

    @property
    def inner_shape(self):
        return self._inner_shape

    def __repr__(self):
        args = [repr(self._primitive)]
        if self._inner_shape is not None and len(self._inner_shape) > 0:
            args.append("inner_shape=" + repr(self._inner_shape))
        args += self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def _tolist_part(self, verbose=True, toplevel=False):
        if toplevel:
            return self._primitive
        out = {}
        out["class"] = "NumpyArray"
        out["primitive"] = self._primitive
        if verbose:
            out["inner_shape"] = [] if self._inner_shape is None else self._inner_shape
        else:
            if self._inner_shape is not None and len(self._inner_shape) > 0:
                out["inner_shape"] = self._inner_shape
        return out
