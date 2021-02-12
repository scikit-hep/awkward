import pytest

import awkward as ak

from generated_parser import Lark_StandAlone, Transformer


class TreeToJson(Transformer):
    def string(self, s):
        (s,) = s
        return s[1:-1]

    def number(self, n):
        (n,) = n
        if "." in n:
            return float(n)
        else:
            return int(n)

    list_obj = list
    pair = tuple
    dict_obj = dict

    null = lambda self, _: None
    true = lambda self, _: True
    false = lambda self, _: False


def toast(ptnode):
    if ptnode.__class__.__name__ == "Token":
        return ptnode.value
    elif ptnode.data == "start":
        return toast(ptnode.children[0])
    elif ptnode.data == "input":
        assert len(ptnode.children) == 1
        return toast(ptnode.children[0])
    elif ptnode.data == "primitive":
        if len(ptnode.children) == 1:
            return ak.types.PrimitiveType(toast(ptnode.children[0]))
        elif len(ptnode.children) == 2:
            return ak.types.PrimitiveType(
                toast(ptnode.children[0]), toast(ptnode.children[1])
            )
        else:
            raise Exception("Unhandled PrimitiveType node")
    elif ptnode.data == "unknown":
        if len(ptnode.children) == 0:
            return ak.types.UnknownType()
        elif len(ptnode.children) == 1:
            return ak.types.UnknownType(parameters=toast(ptnode.children[0]))
        else:
            raise Exception("Unhandled UnknownType node")
    elif ptnode.data == "listtype":
        return toast(ptnode.children[0])
    elif ptnode.data == "list_single":
        return ak.types.ListType(toast(ptnode.children[0]))
    elif ptnode.data == "list_parm":
        return ak.types.ListType(toast(ptnode.children[0]), toast(ptnode.children[1]))
    elif ptnode.data == "uniontype":
        return toast(ptnode.children[0])
    elif ptnode.data == "union_single":
        content_list = []
        for node in ptnode.children:
            content_list.append(toast(node))
        return ak.types.UnionType(content_list)
    elif ptnode.data == "union_parm":
        content_list = []
        for node in ptnode.children[:-1]:
            content_list.append(toast(node))
        return ak.types.UnionType(content_list, toast(ptnode.children[-1]))
    elif ptnode.data == "optiontype":
        if len(ptnode.children) == 1:
            return ak.types.OptionType(toast(ptnode.children[0]))
        elif len(ptnode.children) == 2:
            return ak.types.OptionType(
                toast(ptnode.children[0]), parameters=toast(ptnode.children[1])
            )
        else:
            raise Exception("Unhandled OptionType node")
    elif ptnode.data == "record":
        return toast(ptnode.children[0])
    elif ptnode.data == "record_tuple":
        content_list = []
        for node in ptnode.children:
            content_list.append(toast(node))
        return ak.types.RecordType(tuple(content_list))
    elif ptnode.data == "record_dict":
        content_types = []
        content_keys = []
        for i in range(0, len(ptnode.children), 2):
            content_keys.append(ptnode.children[i])
            content_types.append(toast(ptnode.children[i + 1]))
        return ak.types.RecordType(content_types, content_keys)
    elif ptnode.data == "record_tuple_param":
        content_list = []
        for node in ptnode.children[:-1]:
            content_list.append(toast(node))
        return ak.types.RecordType(
            tuple(content_list), parameters=toast(ptnode.children[-1])
        )
    elif ptnode.data == "record_struct":
        content_list = []
        content_keys = []
        for node in ptnode.children[:-1]:
            if isinstance(node, str):
                content_keys.append(node)
            else:
                content_list.append(toast(node))
        return ak.types.RecordType(
            tuple(content_list),
            keys=content_keys,
            parameters=toast(ptnode.children[-1]),
        )
    elif ptnode.data == "regular":
        assert (len(ptnode.children)) == 1
        return toast(ptnode.children[0])
    elif ptnode.data == "regular_inparm":
        assert len(ptnode.children) == 2
        return ak.types.RegularType(toast(ptnode.children[1]), ptnode.children[0])
    elif ptnode.data == "regular_outparm":
        assert len(ptnode.children) == 3
        return ak.types.RegularType(
            toast(ptnode.children[1]), ptnode.children[0], toast(ptnode.children[2])
        )
    elif ptnode.data == "def_option":
        assert len(ptnode.children) == 1
        return ptnode.children[0]
    elif ptnode.data == "options":
        assert len(ptnode.children) == 1
        return toast(ptnode.children[0])
    else:
        raise Exception("Unhandled node")


def test_primitive_1():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = "int64"
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.PrimitiveType)
    assert (str(parsedtype)) == text


def test_primitive_2():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = 'int64[parameters={"wonky": ["parameter", 3.14]}]'
    # print(test.parse(text).pretty())
    parsedtype = toast(test.parse(text))
    # print(parsedtype)
    # print(type(parsedtype))
    assert isinstance(parsedtype, ak.types.PrimitiveType)
    assert (str(parsedtype)) == text


def test_unknown_1():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = "unknown"
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.UnknownType)
    assert (str(parsedtype)) == text


def test_unknown_2():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = 'unknown[parameters={"wonky": ["parameter", 3.14]}]'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.UnknownType)
    assert str(parsedtype) == text


def test_record_tuple_1():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = "(int64)"
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_tuple_2():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = '(int64[parameters={"wonky": ["bla", 1, 2]}])'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_tuple_3():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = '(int64, int64[parameters={"wonky": ["bla", 1, 2]}])'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_dict_1():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = '{"1": int64}'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_dict_2():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = '{"bla": int64[parameters={"wonky": ["bla", 1, 2]}]}'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_dict_3():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = '{"bla": int64[parameters={"wonky": ["bla", 1, 2]}], "foo": int64}'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_parmtuple_1():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = 'tuple[[int64[parameters={"xkcd": [11, 12, 13]}]], parameters={"wonky": ["bla", 1, 2]}]'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_parmtuple_2():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = 'tuple[[int64, int64], parameters={"wonky": ["bla", 1, 2]}]'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_struct_1():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = 'struct[["1"], [int64[parameters={"xkcd": [11, 12, 13]}]], parameters={"wonky": ["bla", 1, 2]}]'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_struct_2():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = 'struct[["1", "2"], [int64[parameters={"xkcd": [11, 12, 13]}], int64], parameters={"wonky": ["bla", 1, 2]}]'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_option_numpy_1():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = "?int64"
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_option_numpy_2():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = '?int64[parameters={"wonky": [1, 2, 3]}]'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_option_numpy_1_parm():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = 'option[int64, parameters={"foo": "bar"}]'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_option_numpy_2_parm():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = 'option[int64[parameters={"wonky": [1, 2]}], parameters={"foo": "bar"}]'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_option_unknown_1():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = "?unknown"
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_option_unknown_2():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = '?unknown[parameters={"foo": "bar"}]'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_option_unknown_1_parm():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = 'option[unknown, parameters={"foo": "bar"}]'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_option_unknown_2_parm():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = 'option[unknown, parameters={"foo": "bar"}]'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_regular_numpy_1():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = "5 * int64"
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.RegularType)
    assert str(parsedtype) == text


def test_regular_numpy_2():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = '5 * int64[parameters={"bar": "foo"}]'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.RegularType)
    assert str(parsedtype) == text


def test_regular_numpy_2_parm():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = '[0 * int64[parameters={"foo": "bar"}], parameters={"bla": "bloop"}]'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.RegularType)
    assert str(parsedtype) == text


def test_regular_unknown_1_parm():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = '[0 * unknown, parameters={"foo": "bar"}]'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.RegularType)
    assert str(parsedtype) == text


def test_list_numpy_1():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = "var * float64"
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.ListType)
    assert str(parsedtype) == text


def test_list_numpy_1_parm():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = '[var * float64[parameters={"wonky": "boop"}], parameters={"foo": "bar"}]'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.ListType)
    assert str(parsedtype) == text


def test_union_numpy_empty_1():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = 'union[float64[parameters={"wonky": "boop"}], unknown]'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.UnionType)
    assert str(parsedtype) == text


def test_union_numpy_empty_1_parm():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = 'union[float64[parameters={"wonky": "boop"}], unknown, parameters={"pratyush": "das"}]'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.UnionType)
    assert str(parsedtype) == text


@pytest.mark.skip(reason="strings not handled yet")
def test_jim1():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.Array([["one", "two", "three"], [], ["four", "five"]]).type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


@pytest.mark.skip(reason="bytestrings not handled yet")
def test_jim2():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.Array([[b"one", b"two", b"three"], [], [b"four", b"five"]]).type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


@pytest.mark.skip(reason="categoricals not handled yet")
def test_jim3():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.to_categorical(ak.Array(["one", "one", "two", "three", "one", "three"])).type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


@pytest.mark.skip(reason="categoricals not handled yet")
def test_jim4():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.to_categorical(ak.Array([1.1, 1.1, 2.2, 3.3, 1.1, 3.3])).type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


@pytest.mark.skip(reason="record names not handled yet")
def test_jim5():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.Array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], with_name="Thingy").type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


@pytest.mark.skip(reason="record names not handled yet")
def test_jim6():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.Array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}], [], [{"x": 3, "y": 3.3}]], with_name="Thingy").type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


@pytest.mark.skip(reason="option-type lists not handled yet")
def test_jim7():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.Array([[1, 2, 3], None, [4, 5]]).type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


def test_jim8():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.with_parameter(ak.Array([[1, 2, 3], [], [4, 5]]), "wonky", "string").type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


def test_jim9():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.with_parameter(ak.Array([[1, 2, 3], [], [4, 5]]), "wonky", {"other": "JSON"}).type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


def test_jim10():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.with_parameter(ak.Array([[1, 2, 3], None, [4, 5]]), "wonky", "string").type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


def test_jim11():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.with_parameter(ak.Array([1, 2, 3, None, 4, 5]), "wonky", "string").type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


def test_jim12():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.with_parameter(ak.Array([1, 2, 3, 4, 5]), "wonky", "string").type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


def test_jim13():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.Array([1, 2, 3, None, 4, 5]).type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


def test_jim14():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.with_parameter(ak.Array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]), "wonky", "string").type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


def test_jim15():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.Array([(1, 1.1), (2, 2.2), (3, 3.3)]).type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


def test_jim16():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.with_parameter(ak.Array([(1, 1.1), (2, 2.2), (3, 3.3)]), "wonky", "string").type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


def test_jim17():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.Array([[(1, 1.1), (2, 2.2)], [], [(3, 3.3)]]).type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


def test_jim18():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.to_regular(ak.Array([[1, 2], [3, 4], [5, 6]])).type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


def test_jim19():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.with_parameter(ak.to_regular(ak.Array([[1, 2], [3, 4], [5, 6]])), "wonky", "string").type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


def test_jim20():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.with_parameter(ak.Array([1, 2, 3, [1], [1, 2], [1, 2, 3]]), "wonky", "string").type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


@pytest.mark.skip(reason="option of union with parameters not handled yet")
def test_jim21():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.with_parameter(ak.Array([1, 2, 3, None, [1], [1, 2], [1, 2, 3]]), "wonky", "string").type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


def test_jim22():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.Array([1, 2, 3, None, [1], [1, 2], [1, 2, 3]]).type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text


def test_jim23():
    test = Lark_StandAlone(transformer=TreeToJson())
    text = str(ak.Array([1, 2, 3, None, [], [], []]).type)
    print(text)
    parsedtype = toast(test.parse(text))
    assert str(parsedtype) == text
