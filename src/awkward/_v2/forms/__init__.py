# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json

import awkward._v2.forms.emptyform  # noqa: F401
import awkward._v2.forms.numpyform  # noqa: F401
import awkward._v2.forms.regularform  # noqa: F401
import awkward._v2.forms.listform  # noqa: F401
import awkward._v2.forms.listoffsetform  # noqa: F401
import awkward._v2.forms.recordform  # noqa: F401
import awkward._v2.forms.indexedform  # noqa: F401
import awkward._v2.forms.indexedoptionform  # noqa: F401
import awkward._v2.forms.bytemaskedform  # noqa: F401
import awkward._v2.forms.bitmaskedform  # noqa: F401
import awkward._v2.forms.unmaskedform  # noqa: F401
import awkward._v2.forms.unionform  # noqa: F401
import awkward._v2.forms.virtualform  # noqa: F401


def from_iter(input):
    has_identities = input["has_identities"] if "has_identities" in input else False
    parameters = input["parameters"] if "parameters" in input else {}
    form_key = input["form_key"] if "form_key" in input else None
    if isinstance(input, str) or "NumpyArray" == input["class"]:
        if isinstance(input, str):
            return awkward._v2.forms.numpyform.NumpyForm(primitive=input)
        else:
            primitive = input["primitive"]
            inner_shape = input["inner_shape"] if "inner_shape" in input else []
            has_identities = (
                input["has_identities"] if "has_identities" in input else False
            )
            parameters = input["parameters"] if "parameters" in input else {}
            form_key = input["form_key"] if "form_key" in input else None
            return awkward._v2.forms.numpyform.NumpyForm(
                primitive, inner_shape, has_identities, parameters, form_key
            )
    elif "EmptyArray" == input["class"]:
        return awkward._v2.forms.emptyform.EmptyForm(
            has_identities, parameters, form_key
        )
    elif "RegularArray" == input["class"]:
        return awkward._v2.forms.regularform.RegularForm(
            content=from_iter(input["content"]),
            size=input["size"],
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )
    elif "ListArray" == input["class"]:
        return awkward._v2.forms.listform.ListForm(
            starts=input["starts"],
            stops=input["stops"],
            content=from_iter(input["content"]),
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )
    elif "ListOffsetArray" == input["class"]:
        return awkward._v2.forms.listoffsetform.ListOffsetForm(
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
        return awkward._v2.forms.recordform.RecordForm(
            contents=contents,
            recordlookup=recordlookup,
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )
    elif "IndexedArray" == input["class"]:
        return awkward._v2.forms.indexedform.IndexedForm(
            index=input["index"],
            content=from_iter(input["content"]),
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )
    elif "IndexedOptionArray" == input["class"]:
        return awkward._v2.forms.indexedoptionform.IndexedOptionForm(
            index=input["index"],
            content=from_iter(input["content"]),
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )
    elif "IndexedOptionArray" == input["class"]:
        return awkward._v2.forms.indexedoptionform.IndexedOptionForm(
            index=input["index"],
            content=from_iter(input["content"]),
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )
    elif "ByteMaskedArray" == input["class"]:
        return awkward._v2.forms.bytemaskedform.ByteMaskedForm(
            mask=input["mask"],
            content=from_iter(input["content"]),
            valid_when=input["valid_when"],
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )
    elif "BitMaskedArray" == input["class"]:
        return awkward._v2.forms.bitmaskedform.BitMaskedForm(
            mask=input["mask"],
            content=from_iter(input["content"]),
            valid_when=input["valid_when"],
            lsb_order=input["lsb_order"],
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )
    elif "UnmaskedArray" == input["class"]:
        return awkward._v2.forms.unmaskedform.UnmaskedForm(
            content=from_iter(input["content"]),
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )
    elif "UnionArray" == input["class"]:
        return awkward._v2.forms.unionform.UnionForm(
            tags=input["tags"],
            index=input["index"],
            contents=[from_iter(content) for content in input["contents"]],
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )
    elif "VirtualArray" == input["class"]:
        return awkward._v2.forms.virtualform.VirtualForm(
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


def numpy_form(input):
    primitive = input["primitive"]
    has_identities = input["has_identities"] if "has_identities" in input else False
    parameters = input["parameters"] if "parameters" in input else {}
    form_key = input["form_key"] if "form_key" in input else None

    return awkward._v2.forms.emptyform.NumpyForm(
        primitive, has_identities, parameters, form_key
    )
