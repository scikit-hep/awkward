import awkward as ak


def test():
    array = ak._v2.Array([{"x": [[1, 2, 3], [4, 5, 6]]}])
    assert array[:, []].fields == ["x"]
    assert ak._v2.to_list(ak._v2.Array([{"x": [[1, 2, 3], [4, 5, 6]]}])[:, []]) == [
        {"x": []}
    ]
