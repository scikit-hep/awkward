# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401
import numpy as np  # noqa: F401

to_list = ak._v2.operations.to_list


def test_axis_wrap_if_negative_record_v2():
    dict_cell_chain_field = {
        "cell1": [
            {
                "locus": "TRA",
                "v_call": "TRAV1",
                "cdr3_length": 15,
            },  # <-- represents one chain
            {"locus": "TRB", "v_call": "TRBV1", "cdr3_length": 12},
        ],
        "cell2": [{"locus": "TRA", "v_call": "TRAV1", "cdr3_length": 13}],
        "cell3": [],
    }

    r = ak._v2.Record(dict_cell_chain_field)

    with pytest.raises(np.AxisError):
        r = ak._v2.operations.to_regular(r, 0)
    r = ak._v2.operations.to_regular(r, 2)

    list_cell_chain_field = [
        [["TRA", "TRAV1", 15], ["TRB", "TRBV1", 12]],
        [["TRA", "TRAV1", 13]],
        [],
    ]

    a = ak._v2.Array(list_cell_chain_field)
    a = ak._v2.operations.to_regular(a, 0)
    a = ak._v2.operations.to_regular(a, 2)


def test_axis_wrap_if_negative_record_v1():
    dict_cell_chain_field = {
        "cell1": [
            {
                "locus": "TRA",
                "v_call": "TRAV1",
                "cdr3_length": 15,
            },  # <-- represents one chain
            {"locus": "TRB", "v_call": "TRBV1", "cdr3_length": 12},
        ],
        "cell2": [{"locus": "TRA", "v_call": "TRAV1", "cdr3_length": 13}],
        "cell3": [],
    }

    r = ak.Record(dict_cell_chain_field)

    r = ak.to_regular(r, 0)
    r = ak.to_regular(r, 2)

    list_cell_chain_field = [
        [["TRA", "TRAV1", 15], ["TRB", "TRBV1", 12]],
        [["TRA", "TRAV1", 13]],
        [],
    ]

    a = ak.Array(list_cell_chain_field)
    a = ak.to_regular(a, 0)
    a = ak.to_regular(a, 2)
