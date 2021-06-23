# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward._v2.forms.emptyform  # noqa: F401
import awkward._v2.forms.numpyform  # noqa: F401
import awkward._v2.forms.regularform  # noqa: F401
import awkward._v2.forms.listform  # noqa: F401

# import awkward._v2.forms.recordform  # noqa: F401
# import awkward._v2.forms.optionform  # noqa: F401
# import awkward._v2.forms.unionform  # noqa: F401
# import awkward._v2.forms.arrayform  # noqa: F401


def from_iter(input):
    has_identities = input["has_identities"] if "has_identities" in input else False
    parameters = input["parameters"] if "parameters" in input else {}
    form_key = input["form_key"] if "form_key" in input else None
    if "EmptyArray" == input["class"]:
        return awkward._v2.forms.emptyform.EmptyForm(
            has_identities, parameters, form_key
        )
    if "RegularArray" == input["class"]:
        return awkward._v2.forms.regularform.RegularForm(
            content=from_iter(input["content"]),
            size=input["size"],
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )
    if "ListArray" == input["class"]:
        return awkward._v2.forms.listform.ListForm(
            starts=input["starts"],
            stops=input["stops"],
            content=from_iter(input["content"]),
            has_identities=has_identities,
            parameters=parameters,
            form_key=form_key,
        )


def numpy_form(input):
    primitive = input["primitive"]
    has_identities = input["has_identities"] if "has_identities" in input else False
    parameters = input["parameters"] if "parameters" in input else {}
    form_key = input["form_key"] if "form_key" in input else None

    return awkward._v2.forms.emptyform.NumpyForm(has_identities, parameters, form_key)
