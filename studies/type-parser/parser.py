import sys

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


def toast(ptnode, high_level, categorical):
    if ptnode.__class__.__name__ == "Token":
        return ptnode.value
    elif ptnode.data == "start":
        return toast(ptnode.children[0], high_level, categorical)
    elif ptnode.data == "input":
        assert len(ptnode.children) == 1
        return toast(ptnode.children[0], high_level, categorical)
    elif ptnode.data == "predefined_typestr":
        if ptnode.children[0] == "string":
            parms = {"__array__": "string"}
            if categorical:
                parms.update({"__categorical__": True})
                categorical = False
            return ak.types.ListType(
                ak.types.PrimitiveType(
                    "uint8", parameters={"__array__": "char"}, typestr="char"
                ),
                parameters=parms,
                typestr="string",
            )
        elif ptnode.children[0] == "char":
            parms = {"__array__": "char"}
            if categorical:
                parms.update({"__categorical__": True})
                categorical = False
            return ak.types.PrimitiveType("uint8", parameters=parms, typestr="char")
        elif ptnode.children[0] == "byte":
            parms = {"__array__": "byte"}
            if categorical:
                parms.update({"__categorical__": True})
                categorical = False
            return ak.types.PrimitiveType("uint8", parameters=parms, typestr="byte")
        elif ptnode.children[0] == "bytes":
            parms = {"__array__": "bytestring"}
            if categorical:
                parms.update({"__categorical__": True})
                categorical = False
            return ak.types.ListType(
                ak.types.PrimitiveType(
                    "uint8", parameters={"__array__": "byte"}, typestr="byte"
                ),
                parameters=parms,
                typestr="bytes",
            )
        else:
            raise Exception("Unhandled typestring {0}".format(ptnode.children[0]))
    elif ptnode.data == "primitive":
        if len(ptnode.children) == 1:
            parms = {}
            if categorical:
                parms.update({"__categorical__": True})
                categorical = False
            return ak.types.PrimitiveType(
                toast(ptnode.children[0], high_level, False), parameters=parms
            )
        elif len(ptnode.children) == 2:
            parms = toast(ptnode.children[1], high_level, False)
            if categorical:
                parms.update({"__categorical__": True})
                categorical = False
            return ak.types.PrimitiveType(
                toast(ptnode.children[0], high_level, categorical),
                parms,
            )
        else:
            raise Exception("Unhandled PrimitiveType node")
    elif ptnode.data == "categories":
        assert high_level == True
        return toast(ptnode.children[0], high_level, True)
    elif ptnode.data == "unknown":
        if len(ptnode.children) == 0:
            parms = {}
            if categorical:
                parms.update({"__categorical__": True})
                categorical = False
            return ak.types.UnknownType(parameters=parms)
        elif len(ptnode.children) == 1:
            parms = toast(ptnode.children[0], high_level, False)
            if categorical:
                parms.update({"__categorical__": True})
                categorical = False
            return ak.types.UnknownType(parameters=parms)
        else:
            raise Exception("Unhandled UnknownType node")
    elif ptnode.data == "listtype":
        return toast(ptnode.children[0], high_level, categorical)
    elif ptnode.data == "list_single":
        parms = {}
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        return ak.types.ListType(
            toast(ptnode.children[0], high_level, False), parameters=parms
        )
    elif ptnode.data == "list_parm":
        parms = toast(ptnode.children[1], high_level, False)
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        return ak.types.ListType(
            toast(ptnode.children[0], high_level, categorical), parms
        )
    elif ptnode.data == "uniontype":
        return toast(ptnode.children[0], high_level, categorical)
    elif ptnode.data == "union_single":
        parms = {}
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        content_list = []
        for node in ptnode.children:
            content_list.append(toast(node, high_level, False))
        return ak.types.UnionType(content_list, parameters=parms)
    elif ptnode.data == "union_parm":
        parms = toast(ptnode.children[-1], high_level, False)
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        content_list = []
        for node in ptnode.children[:-1]:
            content_list.append(toast(node, high_level, False))
        return ak.types.UnionType(content_list, parms)
    elif ptnode.data == "optiontype":
        return toast(ptnode.children[0], high_level, categorical)
    elif ptnode.data == "option_single":
        parms = {}
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        return ak.types.OptionType(
            toast(ptnode.children[0], high_level, False), parameters=parms
        )
    elif ptnode.data == "option_parm":
        parms = toast(ptnode.children[1], high_level, False)
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        return ak.types.OptionType(
            toast(ptnode.children[0], high_level, False),
            parameters=parms,
        )
    elif ptnode.data == "option_highlevel":
        assert high_level == True
        parms = {}
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        return ak.types.OptionType(
            toast(ptnode.children[0], high_level, False), parameters=parms
        )
    elif ptnode.data == "record":
        return toast(ptnode.children[0], high_level, categorical)
    elif ptnode.data == "record_tuple":
        parms = {}
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        content_list = []
        for node in ptnode.children:
            content_list.append(toast(node, high_level, categorical))
        return ak.types.RecordType(tuple(content_list), parameters=parms)
    elif ptnode.data == "record_dict":
        parms = {}
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        content_types = []
        content_keys = []
        for i in range(0, len(ptnode.children), 2):
            content_keys.append(ptnode.children[i])
            content_types.append(toast(ptnode.children[i + 1], high_level, categorical))
        return ak.types.RecordType(content_types, content_keys, parameters=parms)
    elif ptnode.data == "record_tuple_param":
        parms = toast(ptnode.children[-1], high_level, False)
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        content_list = []
        for node in ptnode.children[:-1]:
            content_list.append(toast(node, high_level, False))
        return ak.types.RecordType(tuple(content_list), parameters=parms)
    elif ptnode.data == "record_struct":
        parms = toast(ptnode.children[-1], high_level, False)
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        content_list = []
        content_keys = []
        for node in ptnode.children[:-1]:
            if isinstance(node, str):
                content_keys.append(node)
            else:
                content_list.append(toast(node, high_level, False))
        return ak.types.RecordType(
            tuple(content_list),
            keys=content_keys,
            parameters=parms,
        )
    elif ptnode.data == "record_highlevel":
        assert high_level == True
        parms = {"__record__": ptnode.children[0]}
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        content_list = []
        content_keys = []
        for node in ptnode.children[1:]:
            if isinstance(node, str):
                content_keys.append(node)
            else:
                content_list.append(toast(node, high_level, False))
        return ak.types.RecordType(
            tuple(content_list),
            keys=content_keys,
            parameters=parms,
        )
    elif ptnode.data == "regular":
        assert (len(ptnode.children)) == 1
        return toast(ptnode.children[0], high_level, categorical)
    elif ptnode.data == "regular_inparm":
        assert len(ptnode.children) == 2
        if high_level:
            return ak.types.ArrayType(
                toast(ptnode.children[1], high_level, categorical), ptnode.children[0]
            )
        return ak.types.RegularType(
            toast(ptnode.children[1], high_level, categorical), ptnode.children[0]
        )
    elif ptnode.data == "regular_outparm":
        assert len(ptnode.children) == 3
        parms = toast(ptnode.children[2], high_level, False)
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        return ak.types.RegularType(
            toast(ptnode.children[1], high_level, False),
            ptnode.children[0],
            parms,
        )
    elif ptnode.data == "def_option":
        assert len(ptnode.children) == 1
        return ptnode.children[0]
    elif ptnode.data == "options":
        assert len(ptnode.children) == 1
        return toast(ptnode.children[0], high_level, categorical)
    else:
        raise Exception("Unhandled node")


def deduce_type(typestr, high_level=False):
    parseobj = Lark_StandAlone(transformer=TreeToJson())
    return toast(parseobj.parse(typestr), high_level, False)


def test_primitive_1():
    text = "int64"
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.PrimitiveType)
    assert (str(parsedtype)) == text


def test_primitive_2():
    text = 'int64[parameters={"wonky": ["parameter", 3.14]}]'
    # print(test.parse(text).pretty())
    parsedtype = deduce_type(text)
    # print(parsedtype)
    # print(type(parsedtype))
    assert isinstance(parsedtype, ak.types.PrimitiveType)
    assert (str(parsedtype)) == text


def test_unknown_1():
    text = "unknown"
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.UnknownType)
    assert (str(parsedtype)) == text


def test_unknown_2():
    text = 'unknown[parameters={"wonky": ["parameter", 3.14]}]'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.UnknownType)
    assert str(parsedtype) == text


def test_record_tuple_1():
    text = "(int64)"
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_tuple_2():
    text = '(int64[parameters={"wonky": ["bla", 1, 2]}])'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_tuple_3():
    text = '(int64, int64[parameters={"wonky": ["bla", 1, 2]}])'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_dict_1():
    text = '{"1": int64}'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_dict_2():
    text = '{"bla": int64[parameters={"wonky": ["bla", 1, 2]}]}'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_dict_3():
    text = '{"bla": int64[parameters={"wonky": ["bla", 1, 2]}], "foo": int64}'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_parmtuple_1():
    text = 'tuple[[int64[parameters={"xkcd": [11, 12, 13]}]], parameters={"wonky": ["bla", 1, 2]}]'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_parmtuple_2():
    text = 'tuple[[int64, int64], parameters={"wonky": ["bla", 1, 2]}]'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_struct_1():
    text = 'struct[["1"], [int64[parameters={"xkcd": [11, 12, 13]}]], parameters={"wonky": ["bla", 1, 2]}]'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_record_struct_2():
    text = 'struct[["1", "2"], [int64[parameters={"xkcd": [11, 12, 13]}], int64], parameters={"wonky": ["bla", 1, 2]}]'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text


def test_option_numpy_1():
    text = "?int64"
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_option_numpy_2():
    text = '?int64[parameters={"wonky": [1, 2, 3]}]'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_option_numpy_1_parm():
    text = 'option[int64, parameters={"foo": "bar"}]'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_option_numpy_2_parm():
    text = 'option[int64[parameters={"wonky": [1, 2]}], parameters={"foo": "bar"}]'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_option_unknown_1():
    text = "?unknown"
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_option_unknown_2():
    text = '?unknown[parameters={"foo": "bar"}]'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_option_unknown_1_parm():
    text = 'option[unknown, parameters={"foo": "bar"}]'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_option_unknown_2_parm():
    text = 'option[unknown, parameters={"foo": "bar"}]'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.OptionType)
    assert str(parsedtype) == text


def test_regular_numpy_1():
    text = "5 * int64"
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.RegularType)
    assert str(parsedtype) == text


def test_regular_numpy_2():
    text = '5 * int64[parameters={"bar": "foo"}]'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.RegularType)
    assert str(parsedtype) == text


def test_regular_numpy_2_parm():
    text = '[0 * int64[parameters={"foo": "bar"}], parameters={"bla": "bloop"}]'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.RegularType)
    assert str(parsedtype) == text


def test_regular_unknown_1_parm():
    text = '[0 * unknown, parameters={"foo": "bar"}]'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.RegularType)
    assert str(parsedtype) == text


def test_list_numpy_1():
    text = "var * float64"
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.ListType)
    assert str(parsedtype) == text


def test_list_numpy_1_parm():
    text = '[var * float64[parameters={"wonky": "boop"}], parameters={"foo": "bar"}]'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.ListType)
    assert str(parsedtype) == text


def test_union_numpy_empty_1():
    text = 'union[float64[parameters={"wonky": "boop"}], unknown]'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.UnionType)
    assert str(parsedtype) == text


def test_union_numpy_empty_1_parm():
    text = 'union[float64[parameters={"wonky": "boop"}], unknown, parameters={"pratyush": "das"}]'
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.UnionType)
    assert str(parsedtype) == text


def test_jim1():
    text = str(ak.Array([["one", "two", "three"], [], ["four", "five"]]).type)
    print(text)
    parsedtype = deduce_type(text, True)
    assert isinstance(parsedtype, ak.types.ArrayType)
    assert str(parsedtype) == text


def test_jim2():
    text = str(ak.Array([[b"one", b"two", b"three"], [], [b"four", b"five"]]).type)
    print(text)
    parsedtype = deduce_type(text, True)
    assert isinstance(parsedtype, ak.types.ArrayType)
    assert str(parsedtype) == text


def test_jim3():
    text = str(
        ak.to_categorical(ak.Array(["one", "one", "two", "three", "one", "three"])).type
    )
    print(text)
    parsedtype = deduce_type(text, True)
    assert isinstance(parsedtype, ak.types.ArrayType)
    assert str(parsedtype) == text


def test_jim4():
    text = str(ak.to_categorical(ak.Array([1.1, 1.1, 2.2, 3.3, 1.1, 3.3])).type)
    print(text)
    parsedtype = deduce_type(text, True)
    assert isinstance(parsedtype, ak.types.ArrayType)
    assert str(parsedtype) == text


def test_jim5():
    text = str(
        ak.Array(
            [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
            with_name="Thingy",
        ).type
    )
    print(text)
    parsedtype = deduce_type(text, True)
    assert isinstance(parsedtype, ak.types.ArrayType)
    assert str(parsedtype) == text


def test_jim6():
    text = str(
        ak.Array(
            [[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}], [], [{"x": 3, "y": 3.3}]],
            with_name="Thingy",
        ).type
    )
    print(text)
    parsedtype = deduce_type(text, True)
    assert isinstance(parsedtype, ak.types.ArrayType)
    assert str(parsedtype) == text


def test_jim7():
    text = str(ak.Array([[1, 2, 3], None, [4, 5]]).type)
    print(text)
    parsedtype = deduce_type(text, True)
    assert isinstance(parsedtype, ak.types.ArrayType)
    assert str(parsedtype) == text


def test_jim8():
    text = str(
        ak.with_parameter(ak.Array([[1, 2, 3], [], [4, 5]]), "wonky", "string").type
    )
    print(text)
    parsedtype = deduce_type(text)
    assert str(parsedtype) == text


def test_jim9():
    text = str(
        ak.with_parameter(
            ak.Array([[1, 2, 3], [], [4, 5]]), "wonky", {"other": "JSON"}
        ).type
    )
    print(text)
    parsedtype = deduce_type(text)
    assert str(parsedtype) == text


def test_jim10():
    text = str(
        ak.with_parameter(ak.Array([[1, 2, 3], None, [4, 5]]), "wonky", "string").type
    )
    print(text)
    parsedtype = deduce_type(text)
    assert str(parsedtype) == text


def test_jim11():
    text = str(
        ak.with_parameter(ak.Array([1, 2, 3, None, 4, 5]), "wonky", "string").type
    )
    print(text)
    parsedtype = deduce_type(text)
    assert str(parsedtype) == text


def test_jim12():
    text = str(ak.with_parameter(ak.Array([1, 2, 3, 4, 5]), "wonky", "string").type)
    print(text)
    parsedtype = deduce_type(text)
    assert str(parsedtype) == text


def test_jim13():
    text = str(ak.Array([1, 2, 3, None, 4, 5]).type)
    print(text)
    parsedtype = deduce_type(text)
    assert str(parsedtype) == text


def test_jim14():
    text = str(
        ak.with_parameter(
            ak.Array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]),
            "wonky",
            "string",
        ).type
    )
    print(text)
    parsedtype = deduce_type(text)
    assert str(parsedtype) == text


def test_jim15():
    text = str(ak.Array([(1, 1.1), (2, 2.2), (3, 3.3)]).type)
    print(text)
    parsedtype = deduce_type(text)
    assert str(parsedtype) == text


def test_jim16():
    text = str(
        ak.with_parameter(
            ak.Array([(1, 1.1), (2, 2.2), (3, 3.3)]), "wonky", "string"
        ).type
    )
    print(text)
    parsedtype = deduce_type(text)
    assert str(parsedtype) == text


def test_jim17():
    text = str(ak.Array([[(1, 1.1), (2, 2.2)], [], [(3, 3.3)]]).type)
    print(text)
    parsedtype = deduce_type(text)
    assert str(parsedtype) == text


def test_jim18():
    text = str(ak.to_regular(ak.Array([[1, 2], [3, 4], [5, 6]])).type)
    print(text)
    parsedtype = deduce_type(text)
    assert str(parsedtype) == text


def test_jim19():
    text = str(
        ak.with_parameter(
            ak.to_regular(ak.Array([[1, 2], [3, 4], [5, 6]])), "wonky", "string"
        ).type
    )
    print(text)
    parsedtype = deduce_type(text)
    assert str(parsedtype) == text


def test_jim20():
    text = str(
        ak.with_parameter(
            ak.Array([1, 2, 3, [1], [1, 2], [1, 2, 3]]), "wonky", "string"
        ).type
    )
    print(text)
    parsedtype = deduce_type(text)
    assert str(parsedtype) == text


def test_jim21():
    text = str(
        ak.with_parameter(
            ak.Array([1, 2, 3, None, [1], [1, 2], [1, 2, 3]]), "wonky", "string"
        ).type
    )
    print(text)
    parsedtype = deduce_type(text)
    assert str(parsedtype) == text


def test_jim22():
    text = str(ak.Array([1, 2, 3, None, [1], [1, 2], [1, 2, 3]]).type)
    print(text)
    parsedtype = deduce_type(text)
    assert str(parsedtype) == text


def test_jim23():
    text = str(ak.Array([1, 2, 3, None, [], [], []]).type)
    print(text)
    parsedtype = deduce_type(text)
    assert str(parsedtype) == text


def test_string():
    text = "string"
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.ListType)
    assert str(parsedtype) == text


def test_hardcoded():
    text = "var * string"
    parsedtype = deduce_type(text)
    assert isinstance(parsedtype, ak.types.ListType)
    assert str(parsedtype) == text


def test_record_highlevel():
    text = 'Thingy["x": int64, "y": float64]'
    parsedtype = deduce_type(text, True)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert str(parsedtype) == text
