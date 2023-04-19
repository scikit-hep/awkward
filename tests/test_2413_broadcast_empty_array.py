# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import awkward as ak


def test_broadcast_arrays():
    x = ak.Array([])
    assert x.layout.is_unknown
    y = ak.Array([1])
    u, v = ak.broadcast_arrays(x, y)

    assert len(u) == len(v) == 0
    assert ak.type(u) == ak.types.ArrayType(ak.types.UnknownType(), 0)
    assert ak.type(v) == ak.types.ArrayType(ak.types.NumpyType("int64"), 0)


def test_where_unknown_condition():
    unknown = ak.Array([])
    x = ak.Array(["x", "y", "z"])
    y = ak.Array([[1, 2, 3], [4], [5, 6, 7, 8]])
    result = ak.where(unknown, x, y)
    assert len(result) == 0
    assert result.type == ak.types.ArrayType(
        ak.types.UnionType(
            [
                ak.types.ListType(
                    ak.types.NumpyType(
                        "uint8", parameters={"__array__": "char"}, typestr="char"
                    ),
                    parameters={"__array__": "string"},
                    typestr="string",
                ),
                ak.types.ListType(ak.types.NumpyType("int64")),
            ]
        ),
        0,
    )


def test_where_unknown_array():
    unknown = ak.Array([])
    cond = ak.Array([True, False, False])
    y = ak.Array([[1, 2, 3], [4], [5, 6, 7, 8]])
    result = ak.where(cond, unknown, y)
    assert result.type == ak.types.ArrayType(
        ak.types.ListType(ak.types.NumpyType("int64")), 0
    )
