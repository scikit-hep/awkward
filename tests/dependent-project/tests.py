# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import dependent

def test_add():
    assert dependent.add(12, 34) == 46
