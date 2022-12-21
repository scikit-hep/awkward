# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import copy

import numpy as np

import awkward as ak


def test_prototypical_example():
    array = ak.Array(
        {
            "Muon_pt": [[1.1, 2.2, 3.3], [], [4.4, 5.5]],
            "Muon_phi": [[1.1, 2.2, 3.3], [], [4.4, 5.5]],
            "Muon_eta": [[1.1, 2.2, 3.3], [], [4.4, 5.5]],
            "Muon_mass": [[1.1, 2.2, 3.3], [], [4.4, 5.5]],
        }
    )
    layout_tt1 = array.layout.to_typetracer()

    # deepcopy not actually needed; layout_tt1.form produces a new Form each time, but be defensive
    labeled_form = copy.deepcopy(layout_tt1.form)

    # in-place assignment of form_keys with relevant names is probably the most convenient interface
    labeled_form.form_key = "record-1"
    labeled_form.contents[0].form_key = "listoffset-1"
    labeled_form.contents[0].content.form_key = "numpy-pt"
    labeled_form.contents[1].form_key = "listoffset-2"
    labeled_form.contents[1].content.form_key = "numpy-phi"
    labeled_form.contents[2].form_key = "listoffset-3"
    labeled_form.contents[2].content.form_key = "numpy-eta"
    labeled_form.contents[3].form_key = "listoffset-4"
    labeled_form.contents[3].content.form_key = "numpy-mass"

    layout_tt2, report = ak._typetracer.typetracer_with_report(labeled_form)
    original = ak.Array(layout_tt2)

    restructured = ak.zip(
        {
            "muons": ak.zip(
                {
                    "pt": original["Muon_pt"],
                    "phi": original["Muon_phi"],
                    "eta": original["Muon_eta"],
                    "mass": original["Muon_mass"],
                }
            )
        }
    )

    # the listoffsets, but not the numpys, have been touched because of broadcasting
    assert report.data_touched == [
        "listoffset-1",
        "listoffset-2",
        "listoffset-3",
        "listoffset-4",
    ]

    pz = restructured.muons.pt * np.sinh(restructured.muons.eta)  # noqa: F841

    # order is preserved: numpy-eta is used before numpy-pt (may or may not be important)
    assert report.data_touched == [
        "listoffset-1",
        "listoffset-2",
        "listoffset-3",
        "listoffset-4",
        "numpy-eta",
        "numpy-pt",
    ]

    # slices are views, so they shouldn't trigger data access
    sliced = restructured.muons[:1]  # noqa: F841
    assert report.data_touched == [
        "listoffset-1",
        "listoffset-2",
        "listoffset-3",
        "listoffset-4",
        "numpy-eta",
        "numpy-pt",
    ]

    print(restructured.muons.mass)  # noqa: T201
    assert report.data_touched == [
        "listoffset-1",
        "listoffset-2",
        "listoffset-3",
        "listoffset-4",
        "numpy-eta",
        "numpy-pt",
        "numpy-mass",
    ]
