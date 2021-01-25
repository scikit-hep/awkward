import awkward as ak

from generated_parser import Lark_StandAlone, Transformer


class TypeParser(object):
    def __init__(self):
        self.arraytype = "Unknown"


def rettype(infocls):
    if infocls.arraytype == "Primitive":
        assert hasattr(infocls, "typename")
        assert hasattr(infocls, "json")
        return ak.types.PrimitiveType(infocls.typename, infocls.json)
    elif infocls.arraytype == "Unknown":
        return ak.types.UnknownType()


class TreeToJson(Transformer):
    def string(self, s):
        (s,) = s
        return s[1:-1]

    def number(self, n):
        (n,) = n
        return float(n)

    list = list
    pair = tuple
    dict = dict

    null = lambda self, _: None
    true = lambda self, _: True
    false = lambda self, _: False


def toast(ptnode, infocls):
    if ptnode.__class__.__name__ == "Token":
        if ptnode.type == "TYPE":
            setattr(infocls, "typename", ptnode.value)
            infocls.arraytype = "Primitive"
        elif ptnode.type == "UNKNOWNTYPE":
            infocls.arraytype = "Unknown"
    elif ptnode.data == "start":
        toast(ptnode.children[0], infocls)
    elif ptnode.data == "input":
        for node in ptnode.children:
            toast(node, infocls)
    elif ptnode.data == "primitiveoption":
        assert len(ptnode.children) == 1
        setattr(infocls, "json", ptnode.children[0])
    elif ptnode.data == "options":
        assert len(ptnode.children) == 1
        toast(ptnode.children[0], infocls)


if __name__ == "__main__":
    test = Lark_StandAlone(transformer=TreeToJson())
    text = 'int64[parameters={"wonky": ["parameter", 3.14]}]'
    print(test.parse(text).pretty())
    tp = TypeParser()
    toast(test.parse(text), tp)
    parsedtype = rettype(tp)
    print(parsedtype)
    print(type(parsedtype))
    text = "unknown"
    tp = TypeParser()
    toast(test.parse(text), tp)
    parsedtype = rettype(tp)
    print(parsedtype)
    print(type(parsedtype))
