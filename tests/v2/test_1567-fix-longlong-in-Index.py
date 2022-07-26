# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401
import skhep_testdata # noqa: F401
import uproot # noqa: F401

to_list = ak._v2.operations.to_list


def test_1417issue_is_none_check_axis():
    tree = uproot.open(skhep_testdata.data_path("uproot-issue413.root"))["mytree"]
    assert ak._v2.to_list(tree["Str"].array()) == [
            "evt-0",
            "evt-1",
            "evt-2",
            "evt-3",
            "evt-4",
        ]