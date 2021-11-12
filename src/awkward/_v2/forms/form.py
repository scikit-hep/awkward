# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def from_iter(input):
    if input is None:
        return None

    if ak._util.isstr(input):
        return ak._v2.forms.numpyform.NumpyForm(primitive=input)

    assert isinstance(input, dict)
    has_identifier = input.get("has_identifier", input.get("has_identities", False))
    parameters = input.get("parameters", None)
    form_key = input.get("form_key", None)

    if input["class"] == "NumpyArray":
        primitive = input["primitive"]
        inner_shape = input.get("inner_shape", [])
        return ak._v2.forms.numpyform.NumpyForm(
            primitive, inner_shape, has_identifier, parameters, form_key
        )

    elif input["class"] == "EmptyArray":
        return ak._v2.forms.emptyform.EmptyForm(has_identifier, parameters, form_key)

    elif input["class"] == "RegularArray":
        return ak._v2.forms.regularform.RegularForm(
            content=from_iter(input["content"]),
            size=input["size"],
            has_identifier=has_identifier,
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] in ("ListArray", "ListArray32", "ListArrayU32", "ListArray64"):
        return ak._v2.forms.listform.ListForm(
            starts=input["starts"],
            stops=input["stops"],
            content=from_iter(input["content"]),
            has_identifier=has_identifier,
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] in (
        "ListOffsetArray",
        "ListOffsetArray32",
        "ListOffsetArrayU32",
        "ListOffsetArray64",
    ):
        return ak._v2.forms.listoffsetform.ListOffsetForm(
            offsets=input["offsets"],
            content=from_iter(input["content"]),
            has_identifier=has_identifier,
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] == "RecordArray":
        if isinstance(input["contents"], dict):
            contents = []
            fields = []
            for key, content in input["contents"].items():
                contents.append(from_iter(content))
                fields.append(key)
        else:
            contents = [from_iter(content) for content in input["contents"]]
            fields = None
        return ak._v2.forms.recordform.RecordForm(
            contents=contents,
            fields=fields,
            has_identifier=has_identifier,
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] in (
        "IndexedArray",
        "IndexedArray32",
        "IndexedArrayU32",
        "IndexedArray64",
    ):
        return ak._v2.forms.indexedform.IndexedForm(
            index=input["index"],
            content=from_iter(input["content"]),
            has_identifier=has_identifier,
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] in (
        "IndexedOptionArray",
        "IndexedOptionArray32",
        "IndexedOptionArray64",
    ):
        return ak._v2.forms.indexedoptionform.IndexedOptionForm(
            index=input["index"],
            content=from_iter(input["content"]),
            has_identifier=has_identifier,
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] == "ByteMaskedArray":
        return ak._v2.forms.bytemaskedform.ByteMaskedForm(
            mask=input["mask"],
            content=from_iter(input["content"]),
            valid_when=input["valid_when"],
            has_identifier=has_identifier,
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] == "BitMaskedArray":
        return ak._v2.forms.bitmaskedform.BitMaskedForm(
            mask=input["mask"],
            content=from_iter(input["content"]),
            valid_when=input["valid_when"],
            lsb_order=input["lsb_order"],
            has_identifier=has_identifier,
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] == "UnmaskedArray":
        return ak._v2.forms.unmaskedform.UnmaskedForm(
            content=from_iter(input["content"]),
            has_identifier=has_identifier,
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] in (
        "UnionArray",
        "UnionArray8_32",
        "UnionArray8_U32",
        "UnionArray8_64",
    ):
        return ak._v2.forms.unionform.UnionForm(
            tags=input["tags"],
            index=input["index"],
            contents=[from_iter(content) for content in input["contents"]],
            has_identifier=has_identifier,
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] == "VirtualArray":
        raise ValueError("Awkward 1.x VirtualArrays are not supported")

    else:
        raise ValueError(
            "Input class: {0} was not recognised".format(repr(input["class"]))
        )


def from_json(input):
    return from_iter(json.loads(input))


def _parameters_equal(one, two, only_array_record=False):
    if one is None and two is None:
        return True
    elif one is None:
        if only_array_record:
            for key in ("__array__", "__record__"):
                if two.get(key) is not None:
                    return False
            else:
                return True
        else:
            for value in two.values():
                if value is not None:
                    return False
            else:
                return True

    elif two is None:
        if only_array_record:
            for key in ("__array__", "__record__"):
                if one.get(key) is not None:
                    return False
            else:
                return True
        else:
            for value in one.values():
                if value is not None:
                    return False
            else:
                return True

    else:
        if only_array_record:
            keys = ("__array__", "__record__")
        else:
            keys = set(one.keys()).union(two.keys())
        for key in keys:
            if one.get(key) != two.get(key):
                return False
        else:
            return True


def _parameters_update(one, two):
    for k, v in two.items():
        if v is not None:
            one[k] = v


class Form(object):
    def _init(self, has_identifier, parameters, form_key):
        if not isinstance(has_identifier, bool):
            raise TypeError(
                "{0} 'has_identifier' must be of type bool, not {1}".format(
                    type(self).__name__, repr(has_identifier)
                )
            )
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError(
                "{0} 'parameters' must be of type dict or None, not {1}".format(
                    type(self).__name__, repr(parameters)
                )
            )
        if form_key is not None and not ak._util.isstr(form_key):
            raise TypeError(
                "{0} 'form_key' must be of type string or None, not {1}".format(
                    type(self).__name__, repr(form_key)
                )
            )

        self._has_identifier = has_identifier
        self._parameters = parameters
        self._form_key = form_key

    @property
    def has_identifier(self):
        return self._has_identifier

    @property
    def parameters(self):
        if self._parameters is None:
            self._parameters = {}
        return self._parameters

    def parameter(self, key):
        if self._parameters is None:
            return None
        else:
            return self._parameters.get(key)

    @property
    def form_key(self):
        return self._form_key

    def __str__(self):
        return json.dumps(self.tolist(verbose=False), indent=4)

    def tolist(self, verbose=True):
        return self._tolist_part(verbose, toplevel=True)

    def _tolist_extra(self, out, verbose):
        if verbose or self._has_identifier:
            out["has_identifier"] = self._has_identifier
        if verbose or (self._parameters is not None and len(self._parameters) > 0):
            out["parameters"] = self.parameters
        if verbose or self._form_key is not None:
            out["form_key"] = self._form_key
        return out

    def to_json(self):
        return json.dumps(self.tolist(verbose=True))

    def _repr_args(self):
        out = []
        if self._has_identifier is not False:
            out.append("has_identifier=" + repr(self._has_identifier))

        if self._parameters is not None and len(self._parameters) > 0:
            out.append("parameters=" + repr(self._parameters))

        if self._form_key is not None:
            out.append("form_key=" + repr(self._form_key))
        return out

    @property
    def type(self):
        return self._type({})

    def type_from_behavior(self, behavior):
        return self._type(ak._v2._util.typestrs(behavior))

    def simplify_optiontype(self):
        return self

    def simplify_uniontype(self):
        return self
