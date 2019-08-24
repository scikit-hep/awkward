import awkward1

def test_dummy3():
    if awkward1._numba.installed:
        assert awkward1.dummy1(3) == 9

    assert awkward1.dummy3(5) == 25
