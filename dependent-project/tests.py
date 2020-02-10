# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import json

import dependent

import awkward1 as ak

def test_producer():
    complicated_data = [1.1, 2.2, 3.3, [1, 2, 3], [], [4, 5], {"x": 12.3, "y": "wow"}]

    assert ak.tolist(dependent.producer()) == complicated_data

    assert json.loads(dependent.consumer(ak.Array(complicated_data).layout)) == complicated_data
