import awkward as ak


def test_dict():
    array = ak.Array([{"muon": [{"pt": 1}, {"pt": 4}]}])
    ttarray = ak.to_backend(array, "typetracer")

    array["emptydict"] = {}
    ttarray["emptydict"] = {}

    assert ttarray.layout.form == array.layout.form


def test_list():
    array = ak.Array([{"muon": [{"pt": 1}, {"pt": 4}]}])
    ttarray = ak.to_backend(array, "typetracer")

    array["emptylist"] = []
    ttarray["emptylist"] = []

    assert ttarray.layout.form == array.layout.form
