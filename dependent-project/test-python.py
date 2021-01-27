# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import json

import build.dependent

import awkward as ak


def test_producer():
    complicated_data = [1.1, 2.2, 3.3, [1, 2, 3], [], [4, 5], {"x": 12.3, "y": "wow"}]

    assert ak.to_list(build.dependent.producer()) == complicated_data

    assert (
        json.loads(build.dependent.consumer(ak.Array(complicated_data).layout))
        == complicated_data
    )
