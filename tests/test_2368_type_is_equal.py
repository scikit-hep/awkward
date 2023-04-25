# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest

import awkward as ak


@pytest.mark.parametrize(
    "typestr",
    [
        "1 * int64",
        "1 * float32",
        "1 * var * int64",
        "1 * 3 * int64",
        "1 * ?int64",
        "1 * {x: int64, y: int64}",
        "1 * union[float32, string]",
        "1 * unknown",
    ],
)
def test_simple_types(typestr):
    type_ = ak.types.from_datashape(typestr)
    assert type_ == type_
    assert type_.is_equal_to(type_)
    assert type_.is_equal_to(type_, all_parameters=True)


def test_complex_types():
    type_ = ak.types.ArrayType(
        ak.types.ListType(
            ak.types.RegularType(
                ak.types.UnionType(
                    [
                        ak.types.ListType(
                            ak.types.NumpyType(
                                "uint8", parameters={"__array__": "char"}
                            ),
                            parameters={"__array__": "string"},
                        ),
                        ak.types.RecordType(
                            [
                                ak.types.NumpyType(
                                    "int64", parameters={"rectilinear": True}
                                ),
                                ak.types.OptionType(
                                    ak.types.RecordType(
                                        [
                                            ak.types.OptionType(
                                                ak.types.NumpyType("float64")
                                            )
                                        ],
                                        ["z"],
                                    )
                                ),
                            ],
                            ["x", "y"],
                        ),
                    ],
                    parameters={"planets": ["mercury", "venus", "earth", "mars"]},
                ),
                3,
            ),
            parameters={"earth": "not flat"},
        ),
        10,
    )
    assert type_.is_equal_to(type_)
    assert type_.is_equal_to(type_, all_parameters=True)

    type_no_parameters = ak.types.from_datashape(
        "10 * var * 3 * union[string, {x: int64, y: ?{z: ?float64}}]"
    )
    assert type_no_parameters.is_equal_to(type_, all_parameters=False)
    assert not type_no_parameters.is_equal_to(type_, all_parameters=True)


def test_record_tuple():
    record_type = ak.types.from_datashape("10 * var * (int64, int32)")
    tuple_type = ak.types.from_datashape("10 * var * {x: int64, y: int32}")
    assert record_type != tuple_type


def test_record_mixed():
    record = ak.types.from_datashape("10 * var * {x: int64, y: int32}")
    permutation = ak.types.from_datashape("10 * var * {y: int64, x: int32}")
    assert record == permutation
