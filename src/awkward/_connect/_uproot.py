# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# v2: drop, since this functionality will be replaced by AwkwardForth.

from __future__ import absolute_import

import json

# don't import awkward._connect._uproot in awkward/__init__.py!
import uproot

import awkward as ak


def can_optimize(interpretation, form):
    if isinstance(interpretation, uproot.interpretation.objects.AsObjects):
        jsonform = json.loads(form.tojson(verbose=True))
        if (
            jsonform["class"] == "ListOffsetArray64"
            and jsonform["parameters"].get("uproot")
            == {"as": "array", "header": True, "speedbump": False}
            and jsonform["content"]["class"] == "ListOffsetArray64"
            and jsonform["content"]["parameters"].get("uproot")
            == {"as": "vector", "header": False}
            and jsonform["content"]["content"]["class"] == "NumpyArray"
            and jsonform["content"]["content"]["inner_shape"] == []
            and (
                jsonform["content"]["content"].get("primitive") == "float64"
                or jsonform["content"]["content"].get("primitive") == "int32"
            )
        ):
            return True

    return False


def basket_array(form, data, byte_offsets, extra):
    import awkward._io

    # FIXME: uproot_issue_90 is just a placeholder, to show how it would be done

    return awkward._io.uproot_issue_90(
        form,
        ak.layout.NumpyArray(data),
        ak.layout.Index32(byte_offsets),
    )
