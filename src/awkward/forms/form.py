# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import json
from collections.abc import Mapping

import awkward as ak
from awkward import _errors
from awkward.typing import Any

np = ak._nplikes.NumpyMetadata.instance()
numpy_backend = ak._backends.NumpyBackend.instance()


def from_dict(input: dict) -> Form:
    if input is None:
        return None

    if isinstance(input, str):
        return ak.forms.NumpyForm(primitive=input)

    assert isinstance(input, dict)
    parameters = input.get("parameters", None)
    form_key = input.get("form_key", None)

    if input["class"] == "NumpyArray":
        primitive = input["primitive"]
        inner_shape = input.get("inner_shape", [])
        return ak.forms.NumpyForm(
            primitive, inner_shape, parameters=parameters, form_key=form_key
        )

    elif input["class"] == "EmptyArray":
        return ak.forms.EmptyForm(parameters=parameters, form_key=form_key)

    elif input["class"] == "RegularArray":
        return ak.forms.RegularForm(
            content=from_dict(input["content"]),
            size=input["size"],
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] in ("ListArray", "ListArray32", "ListArrayU32", "ListArray64"):
        return ak.forms.ListForm(
            starts=input["starts"],
            stops=input["stops"],
            content=from_dict(input["content"]),
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] in (
        "ListOffsetArray",
        "ListOffsetArray32",
        "ListOffsetArrayU32",
        "ListOffsetArray64",
    ):
        return ak.forms.ListOffsetForm(
            offsets=input["offsets"],
            content=from_dict(input["content"]),
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] == "RecordArray":
        # New serialisation
        if "fields" in input:
            if isinstance(input["contents"], Mapping):
                raise _errors.wrap_error(
                    TypeError("new-style RecordForm contents must not be mappings")
                )
            contents = [from_dict(content) for content in input["contents"]]
            fields = input["fields"]
        # Old style record
        elif isinstance(input["contents"], dict):
            contents = []
            fields = []
            for key, content in input["contents"].items():
                contents.append(from_dict(content))
                fields.append(key)
        # Old style tuple
        else:
            contents = [from_dict(content) for content in input["contents"]]
            fields = None
        return ak.forms.RecordForm(
            contents=contents,
            fields=fields,
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] in (
        "IndexedArray",
        "IndexedArray32",
        "IndexedArrayU32",
        "IndexedArray64",
    ):
        return ak.forms.IndexedForm(
            index=input["index"],
            content=from_dict(input["content"]),
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] in (
        "IndexedOptionArray",
        "IndexedOptionArray32",
        "IndexedOptionArray64",
    ):
        return ak.forms.IndexedOptionForm(
            index=input["index"],
            content=from_dict(input["content"]),
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] == "ByteMaskedArray":
        return ak.forms.ByteMaskedForm(
            mask=input["mask"],
            content=from_dict(input["content"]),
            valid_when=input["valid_when"],
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] == "BitMaskedArray":
        return ak.forms.BitMaskedForm(
            mask=input["mask"],
            content=from_dict(input["content"]),
            valid_when=input["valid_when"],
            lsb_order=input["lsb_order"],
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] == "UnmaskedArray":
        return ak.forms.UnmaskedForm(
            content=from_dict(input["content"]),
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] in (
        "UnionArray",
        "UnionArray8_32",
        "UnionArray8_U32",
        "UnionArray8_64",
    ):
        return ak.forms.UnionForm(
            tags=input["tags"],
            index=input["index"],
            contents=[from_dict(content) for content in input["contents"]],
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] == "VirtualArray":
        raise _errors.wrap_error(
            ValueError("Awkward 1.x VirtualArrays are not supported")
        )

    else:
        raise _errors.wrap_error(
            ValueError(
                "input class: {} was not recognised".format(repr(input["class"]))
            )
        )


def from_json(input: str) -> Form:
    return from_dict(json.loads(input))


def _parameters_equal(one, two, only_array_record=False):
    if one is None and two is None:
        return True
    elif one is None:
        if only_array_record:
            # NB: __categorical__ is currently a type-only parameter, but
            # we check it here as types check this too.
            for key in ("__array__", "__record__", "__categorical__"):
                if two.get(key) is not None:
                    return False
            return True
        else:
            for value in two.values():
                if value is not None:
                    return False
            return True

    elif two is None:
        if only_array_record:
            for key in ("__array__", "__record__", "__categorical__"):
                if one.get(key) is not None:
                    return False
            return True
        else:
            for value in one.values():
                if value is not None:
                    return False
            return True

    else:
        if only_array_record:
            keys = ("__array__", "__record__", "__categorical__")
        else:
            keys = set(one.keys()).union(two.keys())
        for key in keys:
            if one.get(key) != two.get(key):
                return False
        return True


def _parameters_intersect(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    """
    Args:
        left: first parameters mapping
        right: second parameters mapping

    Returns the intersected key-value pairs of `left` and `right` as a dictionary.

    """
    result = {}
    for key in left.keys() & right.keys():
        if left[key] == right[key]:
            result[key] = left[key]
    return result


def _parameters_update(one, two):
    for k, v in two.items():
        if v is not None:
            one[k] = v


def _parameters_is_empty(parameters: dict[str, Any] | None) -> bool:
    """
    Args:
        parameters (dict or None): parameters dictionary, or None

    Return True if the parameters dictionary is considered empty, either because it is
    None, or because it does not have any meaningful (non-None) values; otherwise,
    return False.
    """
    if parameters is None:
        return True

    for item in parameters.values():
        if item is not None:
            return False

    return True


class Form:
    is_numpy = False
    is_unknown = False
    is_list = False
    is_regular = False
    is_option = False
    is_indexed = False
    is_record = False
    is_union = False

    def _init(self, parameters, form_key):
        if parameters is not None and not isinstance(parameters, dict):
            raise _errors.wrap_error(
                TypeError(
                    "{} 'parameters' must be of type dict or None, not {}".format(
                        type(self).__name__, repr(parameters)
                    )
                )
            )
        if form_key is not None and not isinstance(form_key, str):
            raise _errors.wrap_error(
                TypeError(
                    "{} 'form_key' must be of type string or None, not {}".format(
                        type(self).__name__, repr(form_key)
                    )
                )
            )

        self._parameters = parameters
        self._form_key = form_key

    @property
    def parameters(self):
        if self._parameters is None:
            self._parameters = {}
        return self._parameters

    @property
    def is_identity_like(self):
        """Return True if the content or its non-list descendents are an identity"""
        raise _errors.wrap_error(NotImplementedError)

    def parameter(self, key):
        if self._parameters is None:
            return None
        else:
            return self._parameters.get(key)

    def purelist_parameter(self, key):
        raise _errors.wrap_error(NotImplementedError)

    @property
    def purelist_isregular(self):
        raise _errors.wrap_error(NotImplementedError)

    @property
    def purelist_depth(self):
        raise _errors.wrap_error(NotImplementedError)

    @property
    def minmax_depth(self):
        raise _errors.wrap_error(NotImplementedError)

    @property
    def branch_depth(self):
        raise _errors.wrap_error(NotImplementedError)

    @property
    def fields(self):
        raise _errors.wrap_error(NotImplementedError)

    @property
    def is_tuple(self):
        raise _errors.wrap_error(NotImplementedError)

    @property
    def form_key(self):
        return self._form_key

    @form_key.setter
    def form_key(self, value):
        if value is not None and not isinstance(value, str):
            raise ak._errors.wrap_error(TypeError("form_key must be None or a string"))
        self._form_key = value

    def __str__(self):
        return json.dumps(self.to_dict(verbose=False), indent=4)

    def to_dict(self, verbose=True):
        return self._to_dict_part(verbose, toplevel=True)

    def _to_dict_extra(self, out, verbose):
        if verbose or (self._parameters is not None and len(self._parameters) > 0):
            out["parameters"] = self.parameters
        if verbose or self._form_key is not None:
            out["form_key"] = self._form_key
        return out

    def to_json(self):
        return json.dumps(self.to_dict(verbose=True))

    def _repr_args(self):
        out = []
        if self._parameters is not None and len(self._parameters) > 0:
            out.append("parameters=" + repr(self._parameters))
        if self._form_key is not None:
            out.append("form_key=" + repr(self._form_key))
        return out

    @property
    def type(self):
        return self._type({})

    def type_from_behavior(self, behavior):
        return self._type(ak._util.typestrs(behavior))

    def columns(self, list_indicator=None, column_prefix=()):
        output = []
        self._columns(column_prefix, output, list_indicator)
        return output

    def select_columns(self, specifier, expand_braces=True):
        if isinstance(specifier, str):
            specifier = [specifier]

        for item in specifier:
            if not isinstance(item, str):
                raise _errors.wrap_error(
                    TypeError("a column-selection specifier must be a list of strings")
                )

        if expand_braces:
            next_specifier = []
            for item in specifier:
                for result in ak._util.expand_braces(item):
                    next_specifier.append(result)
            specifier = next_specifier

        specifier = [[] if item == "" else item.split(".") for item in set(specifier)]
        matches = [True] * len(specifier)

        output = []
        return self._select_columns(0, specifier, matches, output)

    def column_types(self):
        return self._column_types()

    def _columns(self, path, output, list_indicator):
        raise _errors.wrap_error(NotImplementedError)

    def _select_columns(self, index, specifier, matches, output):
        raise _errors.wrap_error(NotImplementedError)

    def _column_types(self):
        raise _errors.wrap_error(NotImplementedError)

    def _to_dict_part(self, verbose, toplevel):
        raise _errors.wrap_error(NotImplementedError)

    def _type(self, typestrs):
        raise _errors.wrap_error(NotImplementedError)

    def length_zero_array(
        self, *, backend=numpy_backend, highlevel=True, behavior=None
    ):
        return ak.operations.ak_from_buffers._impl(
            form=self,
            length=0,
            container={"": b"\x00\x00\x00\x00\x00\x00\x00\x00"},
            buffer_key="",
            backend=backend,
            highlevel=highlevel,
            behavior=behavior,
            simplify=False,
        )

    def length_one_array(self, *, backend=numpy_backend, highlevel=True, behavior=None):
        # A length-1 array will need at least N bytes, where N is the largest dtype (e.g. 256 bit complex)
        # Similarly, a length-1 array will need no more than 2*N bytes, as all contents need at most two
        # index-types e.g. `ListOffsetArray.offsets` for their various index metadata. Therefore, we
        # create a buffer of this length (2N) and instruct all contents to use it (via `buffer_key=""`).
        # At the same time, with all index-like metadata set to 0, the list types will have zero lengths
        # whilst unions, indexed, and option types will contain a single value.
        return ak.operations.ak_from_buffers._impl(
            form=self,
            length=1,
            container={
                "": b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            },
            buffer_key="",
            backend=backend,
            highlevel=highlevel,
            behavior=behavior,
            simplify=False,
        )
