# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import dependent

import awkward1 as ak

def test_producer():
    assert ak.tolist(dependent.producer()) == [1.1, 2.2, 3.3]
    assert dependent.consumer(ak.Array([1.1, 2.2, 3.3]).layout) == "[1.1,2.2,3.3]"
