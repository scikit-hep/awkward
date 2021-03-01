# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import sys

import awkward as ak

from ak._typeparser._generated_parser import Lark_StandAlone, Transformer

class _TreeToJson(Transformer):
    def string(self, s):
        (s,) = s
        if sys.version_info[0] == 2:
            s = s.encode("utf-8")
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


def _toast(ptnode, high_level, categorical):
    if ptnode.__class__.__name__ == "Token":
        return ptnode.value
    elif ptnode.data == "start":
        return _toast(ptnode.children[0], high_level, categorical)
    elif ptnode.data == "input":
        assert len(ptnode.children) == 1
        return _toast(ptnode.children[0], high_level, categorical)
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
                _toast(ptnode.children[0], high_level, False), parameters=parms
            )
        elif len(ptnode.children) == 2:
            parms = _toast(ptnode.children[1], high_level, False)
            if categorical:
                parms.update({"__categorical__": True})
                categorical = False
            return ak.types.PrimitiveType(
                _toast(ptnode.children[0], high_level, categorical),
                parms,
            )
        else:
            raise Exception("Unhandled PrimitiveType node")
    elif ptnode.data == "categories":
        assert high_level == True
        return _toast(ptnode.children[0], high_level, True)
    elif ptnode.data == "unknown":
        if len(ptnode.children) == 0:
            parms = {}
            if categorical:
                parms.update({"__categorical__": True})
                categorical = False
            return ak.types.UnknownType(parameters=parms)
        elif len(ptnode.children) == 1:
            parms = _toast(ptnode.children[0], high_level, False)
            if categorical:
                parms.update({"__categorical__": True})
                categorical = False
            return ak.types.UnknownType(parameters=parms)
        else:
            raise Exception("Unhandled UnknownType node")
    elif ptnode.data == "listtype":
        return _toast(ptnode.children[0], high_level, categorical)
    elif ptnode.data == "list_single":
        parms = {}
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        return ak.types.ListType(
            _toast(ptnode.children[0], high_level, False), parameters=parms
        )
    elif ptnode.data == "list_parm":
        parms = _toast(ptnode.children[1], high_level, False)
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        return ak.types.ListType(
            _toast(ptnode.children[0], high_level, categorical), parms
        )
    elif ptnode.data == "uniontype":
        return _toast(ptnode.children[0], high_level, categorical)
    elif ptnode.data == "union_single":
        parms = {}
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        content_list = []
        for node in ptnode.children:
            content_list.append(_toast(node, high_level, False))
        return ak.types.UnionType(content_list, parameters=parms)
    elif ptnode.data == "union_parm":
        parms = _toast(ptnode.children[-1], high_level, False)
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        content_list = []
        for node in ptnode.children[:-1]:
            content_list.append(_toast(node, high_level, False))
        return ak.types.UnionType(content_list, parms)
    elif ptnode.data == "optiontype":
        return _toast(ptnode.children[0], high_level, categorical)
    elif ptnode.data == "option_single":
        parms = {}
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        return ak.types.OptionType(
            _toast(ptnode.children[0], high_level, False), parameters=parms
        )
    elif ptnode.data == "option_parm":
        parms = _toast(ptnode.children[1], high_level, False)
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        return ak.types.OptionType(
            _toast(ptnode.children[0], high_level, False),
            parameters=parms,
        )
    elif ptnode.data == "option_highlevel":
        assert high_level == True
        parms = {}
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        return ak.types.OptionType(
            _toast(ptnode.children[0], high_level, False), parameters=parms
        )
    elif ptnode.data == "record":
        return _toast(ptnode.children[0], high_level, categorical)
    elif ptnode.data == "record_tuple":
        parms = {}
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        content_list = []
        for node in ptnode.children:
            content_list.append(_toast(node, high_level, categorical))
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
            content_types.append(
                _toast(ptnode.children[i + 1], high_level, categorical)
            )
        return ak.types.RecordType(content_types, content_keys, parameters=parms)
    elif ptnode.data == "record_tuple_param":
        parms = _toast(ptnode.children[-1], high_level, False)
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        content_list = []
        for node in ptnode.children[:-1]:
            content_list.append(_toast(node, high_level, False))
        return ak.types.RecordType(tuple(content_list), parameters=parms)
    elif ptnode.data == "record_struct":
        parms = _toast(ptnode.children[-1], high_level, False)
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        content_list = []
        content_keys = []
        for node in ptnode.children[:-1]:
            if isinstance(node, str):
                content_keys.append(node)
            else:
                content_list.append(_toast(node, high_level, False))
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
                content_list.append(_toast(node, high_level, False))
        return ak.types.RecordType(
            tuple(content_list),
            keys=content_keys,
            parameters=parms,
        )
    elif ptnode.data == "regular":
        assert (len(ptnode.children)) == 1
        return _toast(ptnode.children[0], high_level, categorical)
    elif ptnode.data == "regular_inparm":
        assert len(ptnode.children) == 2
        if high_level:
            return ak.types.ArrayType(
                _toast(ptnode.children[1], high_level, categorical), ptnode.children[0]
            )
        return ak.types.RegularType(
            _toast(ptnode.children[1], high_level, categorical), ptnode.children[0]
        )
    elif ptnode.data == "regular_outparm":
        assert len(ptnode.children) == 3
        parms = _toast(ptnode.children[2], high_level, False)
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        return ak.types.RegularType(
            _toast(ptnode.children[1], high_level, False),
            ptnode.children[0],
            parms,
        )
    elif ptnode.data == "def_option":
        assert len(ptnode.children) == 1
        return ptnode.children[0]
    elif ptnode.data == "options":
        assert len(ptnode.children) == 1
        return _toast(ptnode.children[0], high_level, categorical)
    else:
        raise Exception("Unhandled node")


def from_datashape(typestr, high_level=False):
    parseobj = Lark_StandAlone(transformer=_TreeToJson())
    return _toast(parseobj.parse(typestr), high_level, False)
