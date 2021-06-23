# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import json, copy

from awkward._v2.forms.form import Form

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()

_primitive_to_dtype = {
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
    _primitive_to_dtype["float16"] = np.dtype(np.float16)
if hasattr(np, "float128"):
    _primitive_to_dtype["float128"] = np.dtype(np.float128)
if hasattr(np, "complex256"):
    _primitive_to_dtype["complex256"] = np.dtype(np.complex256)

_dtype_to_primitive = {}
for primitive, dtype in _primitive_to_dtype.items():
    _dtype_to_primitive[dtype] = primitive


def from_dtype(dtype, parameters={}, inner_shape=[]):
    if "[" in str(dtype):
        dtype = str(dtype).split("[")
        # WIP
        params = copy.deepcopy(parameters)
        params["__unit__"] = dtype[1][:-1]
        dtype = dtype[0]
        return NumpyForm(dtype, parameters=params, inner_shape=inner_shape)
    else:
        print(parameters)
        return NumpyForm(
            _dtype_to_primitive[dtype], parameters=parameters, inner_shape=inner_shape
        )


def from_iter(input):
    if isinstance(input, str):
        return NumpyForm(primitive=input)
    else:
        primitive = input["primitive"]
        inner_shape = input["inner_shape"] if "inner_shape" in input else []
        has_identities = input["has_identities"] if "has_identities" in input else False
        parameters = input["parameters"] if "parameters" in input else {}
        form_key = input["form_key"] if "form_key" in input else None
        return NumpyForm(primitive, inner_shape, has_identities, parameters, form_key)


class NumpyForm(Form):
    def __init__(
        self,
        primitive,
        inner_shape=[],
        has_identities=False,
        parameters={},
        form_key=None,
    ):
        if primitive not in _primitive_to_dtype:
            raise TypeError(
                "{0} 'primitive' must be one of {1}, not {2}".format(
                    type(self).__name__,
                    ", ".join(repr(x) for x in _primitive_to_dtype),
                    repr(primitive),
                )
            )
        if not isinstance(inner_shape, Iterable):
            raise TypeError(
                "{0} 'inner_shape' must be iterable, not {1}".format(
                    type(self).__name__, repr(inner_shape)
                )
            )
        if not isinstance(inner_shape, list):
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

    def __repr__(self):
        args = [repr(self._primitive)]
        if len(self._inner_shape) > 0:
            args.append("inner_shape=" + repr(self._inner_shape))
        args += self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def _tolist_part(self, verbose=True):
        out = {}
        out["class"] = "NumpyArray"
        out["primitive"] = self._primitive
        if verbose:
            out["inner_shape"] = self._inner_shape
        else:
            if len(self._inner_shape) > 0:
                out["inner_shape"] = self._inner_shape
        return out
