# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def from_iter(input):
    has_identities = input["has_identities"] if "has_identities" in input else False
    parameters = input["parameters"] if "parameters" in input else {}
    form_key = input["form_key"] if "form_key" in input else None
    if isinstance(input, str) or "NumpyArray" == input["class"]:
        if isinstance(input, str):
            return ak._v2.forms.numpyform.NumpyForm(primitive=input)
        else:
            primitive = input["primitive"]
            inner_shape = input["inner_shape"] if "inner_shape" in input else []
            has_identities = (
                input["has_identities"] if "has_identities" in input else False
            )
            parameters = input["parameters"] if "parameters" in input else {}
            form_key = input["form_key"] if "form_key" in input else None
            return ak._v2.forms.numpyform.NumpyForm(
                primitive, inner_shape, has_identities, parameters, form_key
            )
    elif "EmptyArray" == input["class"]:
        return ak._v2.forms.emptyform.EmptyForm(has_identities, parameters, form_key)
    elif "RegularArray" == input["class"]:
        return ak._v2.forms.regularform.RegularForm(
            content=from_iter(input["content"]),
            size=input["size"],
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )
    elif input["class"] in ("ListArray", "ListArray32", "ListArrayU32", "ListArray64"):
        return ak._v2.forms.listform.ListForm(
            starts=input["starts"],
            stops=input["stops"],
            content=from_iter(input["content"]),
            has_identities=has_identities,
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
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )
    elif "RecordArray" == input["class"]:
        recordlookup = input["recordlookup"] if "recordlookup" in input else None
        if isinstance(input["contents"], dict):
            recordlookup = list(input["contents"].keys())
            contents = [
                from_iter(content) for content in list(input["contents"].values())
            ]
        else:
            contents = [from_iter(content) for content in input["contents"]]
        return ak._v2.forms.recordform.RecordForm(
            contents=contents,
            recordlookup=recordlookup,
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )
    elif "IndexedArray" == input["class"]:
        return ak._v2.forms.indexedform.IndexedForm(
            index=input["index"],
            content=from_iter(input["content"]),
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )
    elif "IndexedOptionArray" == input["class"]:
        return ak._v2.forms.indexedoptionform.IndexedOptionForm(
            index=input["index"],
            content=from_iter(input["content"]),
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )
    elif "ByteMaskedArray" == input["class"]:
        return ak._v2.forms.bytemaskedform.ByteMaskedForm(
            mask=input["mask"],
            content=from_iter(input["content"]),
            valid_when=input["valid_when"],
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )
    elif "BitMaskedArray" == input["class"]:
        return ak._v2.forms.bitmaskedform.BitMaskedForm(
            mask=input["mask"],
            content=from_iter(input["content"]),
            valid_when=input["valid_when"],
            lsb_order=input["lsb_order"],
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )
    elif "UnmaskedArray" == input["class"]:
        return ak._v2.forms.unmaskedform.UnmaskedForm(
            content=from_iter(input["content"]),
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )
    elif "UnionArray" == input["class"]:
        return ak._v2.forms.unionform.UnionForm(
            tags=input["tags"],
            index=input["index"],
            contents=[from_iter(content) for content in input["contents"]],
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )
    elif "VirtualArray" == input["class"]:
        return ak._v2.forms.virtualform.VirtualForm(
            form=input["form"],
            has_length=input["has_length"],
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )
    else:
        raise ValueError(
            "Input class: {0} was not recognised".format(repr(input["class"]))
        )


def from_json(input):
    return from_iter(json.loads(input))


def from_numpy(input):
    primitive = input["primitive"]
    has_identities = input["has_identities"] if "has_identities" in input else False
    parameters = input["parameters"] if "parameters" in input else {}
    form_key = input["form_key"] if "form_key" in input else None

    return ak._v2.forms.emptyform.NumpyForm(
        primitive, has_identities, parameters, form_key
    )


class Form(object):
    @property
    def has_identities(self):
        return self._has_identities

    @property
    def parameters(self):
        return self._parameters

    @property
    def form_key(self):
        return self._form_key

    def parameter(self, key):
        if self._parameters is None:
            return None
        else:
            return self._parameters.get(key)

    def __str__(self):
        return json.dumps(self.tolist(verbose=False), indent="    ")

    def tolist(self, verbose=True, toplevel=False):
        if toplevel:
            return self._tolist_part(verbose=verbose, toplevel=toplevel)
        out = self._tolist_part(verbose=verbose)
        if verbose or self._has_identities:
            out["has_identities"] = self._has_identities
        if verbose or self._parameters is not None and len(self._parameters) > 0:
            out["parameters"] = {} if self._parameters is None else self._parameters
        if verbose or self._form_key is not None:
            out["form_key"] = self._form_key
        return out

    def to_json(self):
        return json.dumps(self.tolist())

    def _repr_args(self):
        out = []
        if self._has_identities is not False:
            out.append("has_identities=" + repr(self._has_identities))

        if self._parameters is not None and len(self._parameters) > 0:
            out.append("parameters=" + repr(self._parameters))

        if self._form_key is not None:
            out.append("form_key=" + repr(self._form_key))
        return out
