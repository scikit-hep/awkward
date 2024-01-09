# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# v2: keep this file, but drop the two *_v1 functions

import awkward as ak

from awkward._typeparser.generated_parser import Lark_StandAlone, Transformer


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

    def null(self, s):
        return None

    def true(self, s):
        return True

    def false(self, s):
        return False


def toast_v1(ptnode, highlevel, categorical):
    if ptnode.__class__.__name__ == "Token":
        return ptnode.value

    elif ptnode.data == "start":
        return toast_v1(ptnode.children[0], highlevel, categorical)

    elif ptnode.data == "input":
        assert len(ptnode.children) == 1
        return toast_v1(ptnode.children[0], highlevel, categorical)

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
            raise AssertionError(f"unhandled typestring {ptnode.children[0]}")

    elif ptnode.data == "primitive":
        if len(ptnode.children) == 1:
            parms = {}
            if categorical:
                parms.update({"__categorical__": True})
                categorical = False
            return ak.types.PrimitiveType(
                toast_v1(ptnode.children[0], highlevel, False), parameters=parms
            )
        elif len(ptnode.children) == 2:
            parms = toast_v1(ptnode.children[1], highlevel, False)
            if categorical:
                parms.update({"__categorical__": True})
                categorical = False
            return ak.types.PrimitiveType(
                toast_v1(ptnode.children[0], highlevel, categorical),
                parms,
            )
        else:
            raise AssertionError("unhandled PrimitiveType node")

    elif ptnode.data == "categories":
        assert highlevel is True
        return toast_v1(ptnode.children[0], highlevel, True)

    elif ptnode.data == "unknown":
        if len(ptnode.children) == 0:
            parms = {}
            if categorical:
                parms.update({"__categorical__": True})
                categorical = False
            return ak.types.UnknownType(parameters=parms)
        elif len(ptnode.children) == 1:
            parms = toast_v1(ptnode.children[0], highlevel, False)
            if categorical:
                parms.update({"__categorical__": True})
                categorical = False
            return ak.types.UnknownType(parameters=parms)
        else:
            raise AssertionError("unhandled UnknownType node")

    elif ptnode.data == "listtype":
        return toast_v1(ptnode.children[0], highlevel, categorical)

    elif ptnode.data == "list_single":
        parms = {}
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        return ak.types.ListType(
            toast_v1(ptnode.children[0], highlevel, False), parameters=parms
        )

    elif ptnode.data == "list_parm":
        parms = toast_v1(ptnode.children[1], highlevel, False)
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        return ak.types.ListType(
            toast_v1(ptnode.children[0], highlevel, categorical), parms
        )

    elif ptnode.data == "uniontype":
        return toast_v1(ptnode.children[0], highlevel, categorical)

    elif ptnode.data == "union_single":
        parms = {}
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        content_list = []
        for node in ptnode.children:
            content_list.append(toast_v1(node, highlevel, False))
        return ak.types.UnionType(content_list, parameters=parms)

    elif ptnode.data == "union_parm":
        parms = toast_v1(ptnode.children[-1], highlevel, False)
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        content_list = []
        for node in ptnode.children[:-1]:
            content_list.append(toast_v1(node, highlevel, False))
        return ak.types.UnionType(content_list, parms)

    elif ptnode.data == "optiontype":
        return toast_v1(ptnode.children[0], highlevel, categorical)

    elif ptnode.data == "option_single":
        parms = {}
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        return ak.types.OptionType(
            toast_v1(ptnode.children[0], highlevel, False), parameters=parms
        )

    elif ptnode.data == "option_parm":
        parms = toast_v1(ptnode.children[1], highlevel, False)
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        return ak.types.OptionType(
            toast_v1(ptnode.children[0], highlevel, False),
            parameters=parms,
        )

    elif ptnode.data == "option_highlevel":
        assert highlevel
        parms = {}
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        return ak.types.OptionType(
            toast_v1(ptnode.children[0], highlevel, False), parameters=parms
        )

    elif ptnode.data == "record":
        return toast_v1(ptnode.children[0], highlevel, categorical)

    elif ptnode.data == "record_tuple":
        parms = {}
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        content_list = []
        for node in ptnode.children:
            content_list.append(toast_v1(node, highlevel, categorical))
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
                toast_v1(ptnode.children[i + 1], highlevel, categorical)
            )
        return ak.types.RecordType(content_types, content_keys, parameters=parms)

    elif ptnode.data == "record_tuple_param":
        parms = toast_v1(ptnode.children[-1], highlevel, False)
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        content_list = []
        for node in ptnode.children[:-1]:
            content_list.append(toast_v1(node, highlevel, False))
        return ak.types.RecordType(tuple(content_list), parameters=parms)

    elif ptnode.data == "record_struct":
        parms = toast_v1(ptnode.children[-1], highlevel, False)
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        content_list = []
        content_keys = []
        for node in ptnode.children[:-1]:
            if isinstance(node, str):
                content_keys.append(node)
            else:
                content_list.append(toast_v1(node, highlevel, False))
        return ak.types.RecordType(
            tuple(content_list),
            keys=content_keys,
            parameters=parms,
        )

    elif ptnode.data == "record_highlevel":
        assert highlevel
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
                content_list.append(toast_v1(node, highlevel, False))
        return ak.types.RecordType(
            tuple(content_list),
            keys=content_keys,
            parameters=parms,
        )

    elif ptnode.data == "regular":
        assert (len(ptnode.children)) == 1
        return toast_v1(ptnode.children[0], highlevel, categorical)

    elif ptnode.data == "regular_inparm":
        assert len(ptnode.children) == 2
        if highlevel:
            return ak.types.ArrayType(
                toast_v1(ptnode.children[1], highlevel, categorical), ptnode.children[0]
            )
        return ak.types.RegularType(
            toast_v1(ptnode.children[1], highlevel, categorical), ptnode.children[0]
        )

    elif ptnode.data == "regular_outparm":
        assert len(ptnode.children) == 3
        parms = toast_v1(ptnode.children[2], highlevel, False)
        if categorical:
            parms.update({"__categorical__": True})
            categorical = False
        return ak.types.RegularType(
            toast_v1(ptnode.children[1], highlevel, False),
            ptnode.children[0],
            parms,
        )

    elif ptnode.data == "def_option":
        assert len(ptnode.children) == 1
        return ptnode.children[0]

    elif ptnode.data == "options":
        assert len(ptnode.children) == 1
        return toast_v1(ptnode.children[0], highlevel, categorical)

    else:
        raise AssertionError("unhandled node")


def from_datashape_v1(typestr, highlevel=False):
    parseobj = Lark_StandAlone(transformer=TreeToJson())
    return toast_v1(parseobj.parse(typestr), highlevel, False)
