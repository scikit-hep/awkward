# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def test_ByteMaskedArray():
    record = ak.zip(
        {
            "x": ak.mask(["foo", "bar", "world"], [True, True, False]),
            "y": ak.mask(["do", "re", "mi"], [False, True, True]),
        },
        optiontype_outside_record=True,
    )
    assert record.to_list() == [None, {"x": "bar", "y": "re"}, None]
    assert record.type == ak.types.ArrayType(
        ak.types.OptionType(
            ak.types.RecordType(
                [
                    ak.types.ListType(
                        ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
                        parameters={"__array__": "string"},
                    ),
                    ak.types.ListType(
                        ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
                        parameters={"__array__": "string"},
                    ),
                ],
                ["x", "y"],
            )
        ),
        3,
        None,
    )


def test_IndexedOptionArray():
    record = ak.zip(
        {"x": ["foo", "bar", None], "y": [None, "re", "mi"]},
        optiontype_outside_record=True,
    )
    assert record.to_list() == [None, {"x": "bar", "y": "re"}, None]
    assert record.type == ak.types.ArrayType(
        ak.types.OptionType(
            ak.types.RecordType(
                [
                    ak.types.ListType(
                        ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
                        parameters={"__array__": "string"},
                    ),
                    ak.types.ListType(
                        ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
                        parameters={"__array__": "string"},
                    ),
                ],
                ["x", "y"],
            )
        ),
        3,
        None,
    )
