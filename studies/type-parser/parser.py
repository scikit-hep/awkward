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
