import awkward as ak

from generated_parser import Lark_StandAlone, Transformer


class TypeParser(object):
    def __init__(self):
        self.arraytype = "Unknown"


def rettype(infocls):
    if infocls.arraytype == "Primitive":
        assert hasattr(infocls, "typename")
        if hasattr(infocls, "json"):
            return ak.types.PrimitiveType(infocls.typename, infocls.json)
        return ak.types.PrimitiveType(infocls.typename)
    elif infocls.arraytype == "Unknown":
        if hasattr(infocls, "json"):
            return ak.types.UnknownType(parameters=infocls.json)
        return ak.types.UnknownType()


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


def toast(ptnode, infocls):
    if ptnode.__class__.__name__ == "Token":
        setattr(infocls, "typename", ptnode.value)
    elif ptnode.data == "start":
        toast(ptnode.children[0], infocls)
    elif ptnode.data == "input":
        for node in ptnode.children:
            toast(node, infocls)
    elif ptnode.data == "primitive":
        infocls.arraytype = "Primitive"
        for node in ptnode.children:
            toast(node, infocls)
    elif ptnode.data == "unknown":
        infocls.arraytype = "Unknown"
        for node in ptnode.children:
            toast(node, infocls)
    elif ptnode.data == "def_option":
        assert len(ptnode.children) == 1
        setattr(infocls, "json", ptnode.children[0])
    elif ptnode.data == "options":
        assert len(ptnode.children) == 1
        toast(ptnode.children[0], infocls)
    else:
        raise Exception("Unhandled node")


if __name__ == "__main__":
    test = Lark_StandAlone(transformer=TreeToJson())
    text = "int64"
    tp = TypeParser()
    toast(test.parse(text), tp)
    parsedtype = rettype(tp)
    assert isinstance(parsedtype, ak._ext.PrimitiveType)
    assert parsedtype.dtype == "int64"
    text = 'int64[parameters={"wonky": ["parameter", 3.14]}]'
    # print(test.parse(text).pretty())
    tp = TypeParser()
    toast(test.parse(text), tp)
    parsedtype = rettype(tp)
    # print(parsedtype)
    # print(type(parsedtype))
    assert isinstance(parsedtype, ak._ext.PrimitiveType)
    assert parsedtype.dtype == "int64"
    assert parsedtype.parameters == {'"wonky"': ['"parameter"', 3.14]}
    text = "unknown"
    tp = TypeParser()
    toast(test.parse(text), tp)
    parsedtype = rettype(tp)
    assert isinstance(parsedtype, ak._ext.UnknownType)
    text = 'unknown[parameters={"wonky": ["parameter", 3.14]}]'
    tp = TypeParser()
    toast(test.parse(text), tp)
    parsedtype = rettype(tp)
    assert isinstance(parsedtype, ak._ext.UnknownType)
    assert parsedtype.parameters == {'"wonky"': ['"parameter"', 3.14]}
