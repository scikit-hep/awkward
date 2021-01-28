import awkward as ak

from generated_parser import Lark_StandAlone, Transformer


class TreeToJson(Transformer):
    def number(self, n):
        (n,) = n
        return float(n)

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
        infocls.arraytype = "Record"
        for node in ptnode.children:
            toast(node, infocls)
    elif ptnode.data == "def_option":
        assert len(ptnode.children) == 1
        return ptnode.children[0]
    elif ptnode.data == "options":
        assert len(ptnode.children) == 1
        return toast(ptnode.children[0])
    else:
        raise Exception("Unhandled node")


if __name__ == "__main__":
    test = Lark_StandAlone(transformer=TreeToJson())
    text = "int64"
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.PrimitiveType)
    assert parsedtype.dtype == "int64"
    text = 'int64[parameters={"wonky": ["parameter", 3.14]}]'
    # print(test.parse(text).pretty())
    parsedtype = toast(test.parse(text))
    # print(parsedtype)
    # print(type(parsedtype))
    assert isinstance(parsedtype, ak.types.PrimitiveType)
    assert parsedtype.dtype == "int64"
    assert parsedtype.parameters == {'"wonky"': ['"parameter"', 3.14]}
    text = "unknown"
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.UnknownType)
    text = 'unknown[parameters={"wonky": ["parameter", 3.14]}]'
    parsedtype = toast(test.parse(text))
    assert isinstance(parsedtype, ak.types.UnknownType)
    assert parsedtype.parameters == {'"wonky"': ['"parameter"', 3.14]}
    """
    text = "(int64)"
    parsedtype = toast(test.parse(text), tp)
    assert isinstance(parsedtype, ak.types.RecordType)
    assert (parsedtype.types == (ak.types.PrimitiveType("int64"), ))
    """
