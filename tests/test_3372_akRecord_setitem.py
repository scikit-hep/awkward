# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import awkward as ak


def test():
    rec = ak.Record({"a": 1})
    assert isinstance(rec.layout, ak.record.Record)

    rec["b"] = 2
    assert isinstance(rec.layout, ak.record.Record)
