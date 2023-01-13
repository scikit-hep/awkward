# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401

import awkward as ak
from awkward.types.listtype import ListType
from awkward.types.numpytype import NumpyType
from awkward.types.optiontype import OptionType
from awkward.types.recordtype import RecordType
from awkward.types.regulartype import RegularType
from awkward.types.uniontype import UnionType
from awkward.types.unknowntype import UnknownType


def test_primitive_1():
    text = "int64"
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.NumpyType)
    assert (str(parsedtype)) == text


def test_primitive_2():
    text = 'int64[parameters={"wonky": ["parameter", 3.14]}]'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.NumpyType)
    assert (str(parsedtype)) == text


def test_unknown_1():
    text = "unknown"
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.UnknownType)
    assert (str(parsedtype)) == text


def test_unknown_2():
    text = 'unknown[parameters={"wonky": ["parameter", 3.14]}]'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.UnknownType)
    assert str(parsedtype) == text


def test_record_tuple_1():
    text = "(int64)"
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_tuple_2():
    text = '(int64[parameters={"wonky": ["bla", 1, 2]}])'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_tuple_3():
    text = '(int64, int64[parameters={"wonky": ["bla", 1, 2]}])'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_dict_1():
    text = '{"1": int64}'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_dict_2():
    text = '{bla: int64[parameters={"wonky": ["bla", 1, 2]}]}'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_dict_3():
    text = '{bla: int64[parameters={"wonky": ["bla", 1, 2]}], foo: int64}'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_parmtuple_1():
    text = 'tuple[[int64[parameters={"xkcd": [11, 12, 13]}]], parameters={"wonky": ["bla", 1, 2]}]'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_parmtuple_2():
    text = 'tuple[[int64, int64], parameters={"wonky": ["bla", 1, 2]}]'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_struct_1():
    text = 'struct[{"1": int64[parameters={"xkcd": [11, 12, 13]}]}, parameters={"wonky": ["bla", 1, 2]}]'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_struct_2():
    text = 'struct[{"1": int64[parameters={"xkcd": [11, 12, 13]}], "2": int64}, parameters={"wonky": ["bla", 1, 2]}]'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_option_numpy_1():
    text = "?int64"
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_option_numpy_2():
    text = '?int64[parameters={"wonky": [1, 2, 3]}]'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_option_numpy_1_parm():
    text = 'option[int64, parameters={"foo": "bar"}]'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_option_numpy_2_parm():
    text = 'option[int64[parameters={"wonky": [1, 2]}], parameters={"foo": "bar"}]'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_option_unknown_1():
    text = "?unknown"
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_option_unknown_2():
    text = '?unknown[parameters={"foo": "bar"}]'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_option_unknown_1_parm():
    text = 'option[unknown, parameters={"foo": "bar"}]'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_option_unknown_2_parm():
    text = 'option[unknown, parameters={"foo": "bar"}]'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_regular_numpy_1():
    text = "5 * int64"
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.RegularType)
    assert str(parsedtype) == text


def test_regular_numpy_2():
    text = '5 * int64[parameters={"bar": "foo"}]'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.RegularType)
    assert str(parsedtype) == text


def test_regular_numpy_2_parm():
    text = '[0 * int64[parameters={"foo": "bar"}], parameters={"bla": "bloop"}]'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.RegularType)
    assert str(parsedtype) == text


def test_regular_unknown_1_parm():
    text = '[0 * unknown, parameters={"foo": "bar"}]'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.RegularType)
    assert str(parsedtype) == text


def test_list_numpy_1():
    text = "var * float64"
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.ListType)
    assert str(parsedtype) == text


def test_list_numpy_1_parm():
    text = '[var * float64[parameters={"wonky": "boop"}], parameters={"foo": "bar"}]'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.ListType)
    assert str(parsedtype) == text


def test_union_numpy_empty_1():
    text = 'union[float64[parameters={"wonky": "boop"}], unknown]'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.UnionType)
    assert str(parsedtype) == text


def test_union_numpy_empty_1_parm():
    text = 'union[float64[parameters={"wonky": "boop"}], unknown, parameters={"pratyush": "das"}]'
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.UnionType)
    assert str(parsedtype) == text


def test_arraytype_string():
    text = str(ak.Array([["one", "two", "three"], [], ["four", "five"]]).type)
    parsedtype = ak.types.from_datashape(text, highlevel=True)
    assert isinstance(parsedtype, ak.types.ArrayType)
    assert str(parsedtype) == text


def test_arraytype_bytestring():
    text = str(ak.Array([[b"one", b"two", b"three"], [], [b"four", b"five"]]).type)
    parsedtype = ak.types.from_datashape(text, highlevel=True)
    assert isinstance(parsedtype, ak.types.ArrayType)
    assert str(parsedtype) == text


def test_arraytype_categorical_1():
    text = str(
        ak.operations.ak_to_categorical.to_categorical(
            ak.Array(["one", "one", "two", "three", "one", "three"])
        ).type
    )
    parsedtype = ak.types.from_datashape(text, highlevel=True)
    assert isinstance(parsedtype, ak.types.ArrayType)
    assert str(parsedtype) == text


def test_arraytype_categorical_2():
    text = str(
        ak.operations.ak_to_categorical.to_categorical(
            ak.Array([1.1, 1.1, 2.2, 3.3, 1.1, 3.3])
        ).type
    )
    parsedtype = ak.types.from_datashape(text, highlevel=True)
    assert isinstance(parsedtype, ak.types.ArrayType)
    assert str(parsedtype) == text


def test_arraytype_record_1():
    text = str(
        ak.Array(
            [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
            with_name="Thingy",
        ).type
    )
    parsedtype = ak.types.from_datashape(text, highlevel=True)
    assert isinstance(parsedtype, ak.types.ArrayType)
    assert str(parsedtype) == text


def test_arraytype_record_2():
    text = str(
        ak.Array(
            [[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}], [], [{"x": 3, "y": 3.3}]],
            with_name="Thingy",
        ).type
    )
    parsedtype = ak.types.from_datashape(text, highlevel=True)
    assert isinstance(parsedtype, ak.types.ArrayType)
    assert str(parsedtype) == text


def test_arraytype_1():
    text = str(ak.Array([[1, 2, 3], None, [4, 5]]).type)
    parsedtype = ak.types.from_datashape(text, highlevel=True)
    assert isinstance(parsedtype, ak.types.ArrayType)
    assert str(parsedtype) == text


def test_arraytype_2():
    text = str(
        ak.with_parameter(ak.Array([[1, 2, 3], [], [4, 5]]), "wonky", "string").type
    )
    parsedtype = ak.types.from_datashape(text, highlevel=True)
    assert isinstance(parsedtype, ak.types.ArrayType)
    assert str(parsedtype) == text


def test_arraytype_3():
    text = str(
        ak.with_parameter(
            ak.Array([[1, 2, 3], [], [4, 5]]), "wonky", {"other": "JSON"}
        ).type
    )
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert str(parsedtype) == text


def test_arraytype_4():
    text = str(
        ak.with_parameter(ak.Array([[1, 2, 3], None, [4, 5]]), "wonky", "string").type
    )
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert str(parsedtype) == text


def test_arraytype_5():
    text = str(
        ak.with_parameter(ak.Array([1, 2, 3, None, 4, 5]), "wonky", "string").type
    )
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert str(parsedtype) == text


def test_arraytype_6():
    text = str(ak.with_parameter(ak.Array([1, 2, 3, 4, 5]), "wonky", "string").type)
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert str(parsedtype) == text


def test_arraytype_7():
    text = str(ak.Array([1, 2, 3, None, 4, 5]).type)
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert str(parsedtype) == text


def test_arraytype_8():
    text = str(
        ak.with_parameter(
            ak.Array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]),
            "wonky",
            "string",
        ).type
    )
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert str(parsedtype) == text


def test_arraytype_9():
    text = str(ak.Array([(1, 1.1), (2, 2.2), (3, 3.3)]).type)
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert str(parsedtype) == text


def test_arraytype_10():
    text = str(
        ak.with_parameter(
            ak.Array([(1, 1.1), (2, 2.2), (3, 3.3)]), "wonky", "string"
        ).type
    )
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert str(parsedtype) == text


def test_arraytype_11():
    text = str(ak.Array([[(1, 1.1), (2, 2.2)], [], [(3, 3.3)]]).type)
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert str(parsedtype) == text


def test_arraytype_12():
    text = str(ak.to_regular(ak.Array([[1, 2], [3, 4], [5, 6]])).type)
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert str(parsedtype) == text


def test_arraytype_13():
    text = str(
        ak.with_parameter(
            ak.to_regular(ak.Array([[1, 2], [3, 4], [5, 6]])), "wonky", "string"
        ).type
    )
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert str(parsedtype) == text


def test_arraytype_14():
    text = str(
        ak.with_parameter(
            ak.Array([1, 2, 3, [1], [1, 2], [1, 2, 3]]), "wonky", "string"
        ).type
    )
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert str(parsedtype) == text


def test_arraytype_15():
    text = str(
        ak.with_parameter(
            ak.Array([1, 2, 3, None, [1], [1, 2], [1, 2, 3]]), "wonky", "string"
        ).type
    )
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert str(parsedtype) == text


def test_arraytype_16():
    text = str(ak.Array([1, 2, 3, None, [1], [1, 2], [1, 2, 3]]).type)
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert str(parsedtype) == text


def test_arraytype_17():
    text = str(ak.Array([1, 2, 3, None, [], [], []]).type)
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert str(parsedtype) == text


def test_string():
    text = "string"
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.ListType)
    assert str(parsedtype) == text


def test_hardcoded():
    text = "var * string"
    parsedtype = ak.types.from_datashape(text, highlevel=False)
    assert isinstance(parsedtype, ak.types.ListType)
    assert str(parsedtype) == text


def test_record_highlevel():
    text = "Thingy[x: int64, y: float64]"
    parsedtype = ak.types.from_datashape(text, highlevel=True)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_numpytype_int32():
    t = NumpyType("int32")
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_numpytype_datetime64():
    t = NumpyType("datetime64")
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_numpytype_datetime64_10s():
    t = NumpyType("datetime64[10s]")
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_numpytype_int32_parameter():
    t = NumpyType("int32", parameters={"__array__": "Something"})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_numpytype_datetime64_parameter():
    t = NumpyType("datetime64", parameters={"__array__": "Something"})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_numpytype_datetime64_10s_parameter():
    t = NumpyType("datetime64[10s]", parameters={"__array__": "Something"})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_numpytype_int32_categorical():
    t = NumpyType("int32", parameters={"__categorical__": True})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_numpytype_int32_parameters_categorical():
    t = NumpyType(
        "int32", parameters={"__array__": "Something", "__categorical__": True}
    )
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_unknowntype():
    t = UnknownType()
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_unknowntype_parameter():
    t = UnknownType(parameters={"__array__": "Something"})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_unknowntype_categorical():
    t = UnknownType(parameters={"__categorical__": True})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_unknowntype_categorical_parameter():
    t = UnknownType(parameters={"__array__": "Something", "__categorical__": True})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_regulartype_numpytype():
    t = RegularType(NumpyType("int32"), 5)
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_regulartype_numpytype_parameter():
    t = RegularType(NumpyType("int32"), 5, parameters={"__array__": "Something"})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_regulartype_numpytype_categorical():
    t = RegularType(NumpyType("int32"), 5, parameters={"__categorical__": True})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_regulartype_numpytype_categorical_parameter():
    t = RegularType(
        NumpyType("int32"),
        5,
        parameters={"__categorical__": True, "__array__": "Something"},
    )
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_listtype_numpytype():
    t = ListType(NumpyType("int32"))
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_listtype_numpytype_parameter():
    t = ListType(NumpyType("int32"), parameters={"__array__": "Something"})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_listtype_numpytype_categorical():
    t = ListType(NumpyType("int32"), parameters={"__categorical__": True})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_listtype_numpytype_categorical_parameter():
    t = ListType(
        NumpyType("int32"),
        parameters={"__categorical__": True, "__array__": "Something"},
    )
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_varlen_string():
    t = ListType(
        NumpyType("uint8", parameters={"__array__": "char"}),
        parameters={"__array__": "string"},
    )
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_varlen_bytestring():
    t = ListType(
        NumpyType("uint8", parameters={"__array__": "char"}),
        parameters={"__array__": "bytestring"},
    )
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_fixedlen_string():
    t = RegularType(
        NumpyType("uint8", parameters={"__array__": "char"}),
        5,
        parameters={"__array__": "string"},
    )
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_fixedlen_bytestring():
    t = RegularType(
        NumpyType("uint8", parameters={"__array__": "byte"}),
        5,
        parameters={"__array__": "bytestring"},
    )
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_char():
    t = NumpyType("uint8", parameters={"__array__": "char"})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_byte():
    t = NumpyType("uint8", parameters={"__array__": "byte"})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_optiontype_numpytype_int32():
    t = OptionType(NumpyType("int32"))
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_optiontype_numpytype_int32_parameters():
    t = OptionType(NumpyType("int32"), parameters={"__array__": "Something"})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_optiontype_numpytype_int32_categorical():
    t = OptionType(NumpyType("int32"), parameters={"__categorical__": True})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_optiontype_numpytype_int32_categorical_parameters():
    t = OptionType(
        NumpyType("int32"),
        parameters={"__array__": "Something", "__categorical__": True},
    )
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_option_varlen_string():
    t = OptionType(
        ListType(
            NumpyType("uint8", parameters={"__array__": "char"}),
            parameters={"__array__": "string"},
        )
    )
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_option_varlen_string_parameters():
    t = OptionType(
        ListType(
            NumpyType("uint8", parameters={"__array__": "char"}),
            parameters={"__array__": "string"},
        ),
        parameters={"__array__": "Something"},
    )
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_record_empty():
    t = RecordType([], None)
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_record_fields_empty():
    t = RecordType([], [])
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_record_int32():
    t = RecordType([NumpyType("int32")], None)
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_record_int32_float64():
    t = RecordType([NumpyType("int32"), NumpyType("float64")], None)
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_record_fields_int32():
    t = RecordType([NumpyType("int32")], ["one"])
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_record_fields_int32_float64():
    t = RecordType([NumpyType("int32"), NumpyType("float64")], ["one", "t w o"])
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_record_empty_parameters():
    t = RecordType([], None, parameters={"p": [123]})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_record_fields_empty_parameters():
    t = RecordType([], [], parameters={"p": [123]})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_record_int32_parameters():
    t = RecordType([NumpyType("int32")], None, parameters={"p": [123]})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_record_int32_float64_parameters():
    t = RecordType(
        [NumpyType("int32"), NumpyType("float64")], None, parameters={"p": [123]}
    )
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_record_fields_int32_parameters():
    t = RecordType([NumpyType("int32")], ["one"], parameters={"p": [123]})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_record_fields_int32_float64_parameters():
    t = RecordType(
        [NumpyType("int32"), NumpyType("float64")],
        ["one", "t w o"],
        parameters={"p": [123]},
    )
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_named_record_empty():
    t = RecordType([], None, parameters={"__record__": "Name"})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_named_record_int32():
    t = RecordType([NumpyType("int32")], None, parameters={"__record__": "Name"})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_named_record_int32_float64():
    t = RecordType(
        [NumpyType("int32"), NumpyType("float64")],
        None,
        parameters={"__record__": "Name"},
    )
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_named_record_fields_int32():
    t = RecordType([NumpyType("int32")], ["one"], parameters={"__record__": "Name"})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_named_record_fields_int32_float64():
    t = RecordType(
        [NumpyType("int32"), NumpyType("float64")],
        ["one", "t w o"],
        parameters={"__record__": "Name"},
    )
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_named_record_empty_parameters():
    t = RecordType([], None, parameters={"__record__": "Name", "p": [123]})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_named_record_int32_parameters():
    t = RecordType(
        [NumpyType("int32")], None, parameters={"__record__": "Name", "p": [123]}
    )
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_named_record_int32_float64_parameters():
    t = RecordType(
        [NumpyType("int32"), NumpyType("float64")],
        None,
        parameters={"__record__": "Name", "p": [123]},
    )
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_named_record_fields_int32_parameters():
    t = RecordType(
        [NumpyType("int32")], ["one"], parameters={"__record__": "Name", "p": [123]}
    )
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_named_record_fields_int32_float64_parameters():
    t = RecordType(
        [NumpyType("int32"), NumpyType("float64")],
        ["one", "t w o"],
        parameters={"__record__": "Name", "p": [123]},
    )
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_union_empty():
    t = UnionType([])
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_union_float64():
    t = UnionType([NumpyType("float64")])
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_union_float64_datetime64():
    t = UnionType(
        [NumpyType("float64"), NumpyType("datetime64")],
    )
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_union_float64_parameters():
    t = UnionType([NumpyType("float64")], parameters={"__array__": "Something"})
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)


def test_union_float64_datetime64_parameters():
    t = UnionType(
        [NumpyType("float64"), NumpyType("datetime64")],
        parameters={"__array__": "Something"},
    )
    assert str(ak.types.from_datashape(str(t), highlevel=False)) == str(t)
