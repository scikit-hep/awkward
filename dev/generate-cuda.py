# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import argparse
import ast
import copy
import os
import sys
from collections import OrderedDict

import yaml

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
KERNEL_WHITELIST = [
    "awkward_new_Identities",
    "awkward_Identities32_to_Identities64",
    "awkward_ListOffsetArray_flatten_offsets",
    "awkward_IndexedArray_overlay_mask",
    "awkward_IndexedArray_mask",
    "awkward_ByteMaskedArray_mask",
    "awkward_zero_mask",
    "awkward_RegularArray_compact_offsets",
    "awkward_IndexedArray_fill_count",
    "awkward_UnionArray_fillna",
    "awkward_localindex",
    "awkward_content_reduce_zeroparents_64",
    "awkward_ListOffsetArray_reduce_global_startstop_64",
    "awkward_IndexedArray_reduce_next_fix_offsets_64",
    "awkward_Index_to_Index64",
    "awkward_carry_arange",
    "awkward_index_carry_nocheck",
    "awkward_NumpyArray_contiguous_init",
    "awkward_NumpyArray_getitem_next_array_advanced",
    "awkward_NumpyArray_getitem_next_at",
    "awkward_RegularArray_getitem_next_array_advanced",
    "awkward_ByteMaskedArray_toIndexedOptionArray",
    "awkward_combinations",  # ?
    "awkward_IndexedArray_simplify",
    "awkward_UnionArray_validity",
    "awkward_index_carry",
    "awkward_ByteMaskedArray_getitem_carry",
    "awkward_IndexedArray_validity",
    "awkward_ByteMaskedArray_overlay_mask",
    "awkward_NumpyArray_reduce_mask_ByteMaskedArray_64",
    "awkward_RegularArray_getitem_carry",
    "awkward_NumpyArray_getitem_next_array",
    "awkward_RegularArray_localindex",
    "awkward_NumpyArray_contiguous_next",
    "awkward_NumpyArray_getitem_next_range",
    "awkward_NumpyArray_getitem_next_range_advanced",
    "awkward_RegularArray_getitem_next_range",
    "awkward_RegularArray_getitem_next_range_spreadadvanced",
    "awkward_RegularArray_getitem_next_array",
    "awkward_missing_repeat",
    "awkward_Identities_getitem_carry",
    "awkward_RegularArray_getitem_jagged_expand",
    "awkward_ListArray_getitem_jagged_expand",
    "awkward_ListArray_getitem_next_array",
    "awkward_RegularArray_broadcast_tooffsets",
    "awkward_NumpyArray_fill_tobool",
    "awkward_NumpyArray_reduce_adjust_starts_64",
    "awkward_NumpyArray_reduce_adjust_starts_shifts_64",
    "awkward_regularize_arrayslice",
    "awkward_RegularArray_getitem_next_at",
    # "awkward_ListOffsetArray_compact_offsets", Need to tune tests
    "awkward_BitMaskedArray_to_IndexedOptionArray",
]


def getthread_dim(pos):
    if pos == 0:
        code = "thread_id"
    elif pos == 1:
        code = "thready_dim"
    elif pos == 2:
        code = "threadz_dim"
    else:
        raise Exception("Cannot handle more than triply nested loops")
    return code


def traverse(node, args={}, forvars=[], declared=[]):  # noqa: B006
    if node.__class__.__name__ == "For":
        forvars.append(traverse(node.target, args, [], declared))
        if len(forvars) == 1:
            thread_var = "thread_id"
        elif len(forvars) == 2:
            thread_var = "thready_dim"
        elif len(forvars) == 3:
            thread_var = "threadz_dim"
        else:
            raise Exception("Cannot handle more than triply nested loops")
        if len(node.iter.args) == 1:
            code = f"if ({thread_var} < {traverse(node.iter.args[0])}) {{\n"
        elif len(node.iter.args) == 2:
            code = "if (({0} < {1}) && ({0} >= {2})) {{\n".format(
                thread_var, traverse(node.iter.args[1]), traverse(node.iter.args[0])
            )
        else:
            raise Exception("Unable to handle Python for loops with >2 args")
        for subnode in node.body:
            code += traverse(subnode, args, copy.copy(forvars), declared)
        code += "}\n"
    elif node.__class__.__name__ == "While":
        assert node.test.__class__.__name__ == "Compare"
        assert len(node.test.ops) == 1
        code = f"while ({traverse(node.test)}) {{\n"
        for subnode in node.body:
            code += traverse(subnode, args, copy.copy(forvars), declared)
        code += "}\n"
    elif node.__class__.__name__ == "Raise":
        if sys.version_info < (3, 8):
            code = f'err->str = "{node.exc.args[0].s}";\n'
        else:
            code = f'err->str = "{node.exc.args[0].value}";\n'
        code += "err->filename = FILENAME(__LINE__);\nerr->pass_through=true;\n"
    elif node.__class__.__name__ == "If":
        code = ""
        code += "if ({0}) {{\n".format(
            traverse(node.test, args, copy.copy(forvars), declared)
        )
        tempdeclared = copy.copy(declared)
        for subnode in node.body:
            code += (
                " " * 2
                + traverse(subnode, args, copy.copy(forvars), tempdeclared)
                + "\n"
            )
        code += "} else {\n"
        for subnode in node.orelse:
            code += (
                " " * 2 + traverse(subnode, args, copy.copy(forvars), declared) + "\n"
            )
        code += "}\n"
    elif node.__class__.__name__ == "BoolOp":
        if node.op.__class__.__name__ == "Or":
            operator = "||"
        elif node.op.__class__.__name__ == "And":
            operator = "&&"
        assert len(node.values) == 2
        code = "{} {} {}".format(
            traverse(node.values[0], args, copy.copy(forvars), declared),
            operator,
            traverse(node.values[1], args, copy.copy(forvars), declared),
        )
    elif node.__class__.__name__ == "Name":
        if node.id in forvars:
            code = getthread_dim(forvars.index(node.id))
        else:
            code = node.id
    elif node.__class__.__name__ == "NameConstant":
        if node.value is True:
            code = "true"
        elif node.value is False:
            code = "false"
        else:
            raise Exception(f"Unhandled NameConstant value {node.value}")
    elif node.__class__.__name__ == "Num":
        code = str(node.n)
    elif node.__class__.__name__ == "BinOp":
        left = traverse(node.left, args, copy.copy(forvars), declared)
        right = traverse(node.right, args, copy.copy(forvars), declared)
        if left in forvars:
            left = getthread_dim(forvars.index(left))
        if right in forvars:
            right = getthread_dim(forvars.index(right))
        code = "({} {} {})".format(
            left,
            traverse(node.op, args, copy.copy(forvars), declared),
            right,
        )
    elif node.__class__.__name__ == "UnaryOp":
        if node.op.__class__.__name__ == "USub":
            code = "-{}".format(
                traverse(node.operand, args, copy.copy(forvars), declared)
            )
        elif node.op.__class__.__name__ == "Not":
            code = "!{}".format(
                traverse(node.operand, args, copy.copy(forvars), declared)
            )
        else:
            raise Exception(
                "Unhandled UnaryOp node {}. Please inform the developers.".format(
                    node.op.__class__.__name__
                )
            )
    elif node.__class__.__name__ == "BitOr":
        code = "|"
    elif node.__class__.__name__ == "Sub":
        code = "-"
    elif node.__class__.__name__ == "Add":
        code = "+"
    elif node.__class__.__name__ == "Mult":
        code = "*"
    elif node.__class__.__name__ == "Subscript":
        if node.slice.__class__.__name__ == "Name" and node.slice.id in forvars:
            code = (
                node.value.id + "[" + getthread_dim(forvars.index(node.slice.id)) + "]"
            )
        elif (
            node.slice.__class__.__name__ == "Constant"
            or node.slice.__class__.__name__ == "BinOp"
            or node.slice.__class__.__name__ == "Subscript"
            or node.slice.__class__.__name__ == "Name"
            or node.slice.__class__.__name__ == "Num"
        ) and hasattr(node.value, "id"):
            code = (
                node.value.id
                + "["
                + str(traverse(node.slice, args, copy.copy(forvars), declared))
                + "]"
            )
        elif node.value.__class__.__name__ == "Subscript":
            code = (
                traverse(node.value.value)
                + "["
                + traverse(node.value.slice)
                + "]["
                + traverse(node.slice)
                + "]"
            )
        else:
            code = traverse(node.slice, args, copy.copy(forvars), declared)
    elif node.__class__.__name__ == "Call":
        assert len(node.args) == 1
        if node.func.id == "uint8":
            casttype = "uint8_t"
        else:
            casttype = node.func.id
        code = "({})({})".format(
            casttype, traverse(node.args[0], args, copy.copy(forvars), declared)
        )
    elif node.__class__.__name__ == "BitAnd":
        code = "&"
    elif node.__class__.__name__ == "Constant":
        if node.value is True:
            code = "true"
        elif node.value is False:
            code = "false"
        else:
            code = node.value
    elif node.__class__.__name__ == "Compare":
        if len(node.ops) == 1 and node.ops[0].__class__.__name__ == "Lt":
            code = "({} < {})".format(
                traverse(node.left, args, copy.copy(forvars), declared),
                traverse(node.comparators[0], args, copy.copy(forvars), declared),
            )
        elif len(node.ops) == 1 and node.ops[0].__class__.__name__ == "NotEq":
            code = "({} != {})".format(
                traverse(node.left, args, copy.copy(forvars), declared),
                traverse(node.comparators[0], args, copy.copy(forvars), declared),
            )
        elif len(node.ops) == 1 and node.ops[0].__class__.__name__ == "Eq":
            code = "({} == {})".format(
                traverse(node.left, args, copy.copy(forvars), declared),
                traverse(node.comparators[0], args, copy.copy(forvars), declared),
            )
        elif len(node.ops) == 1 and node.ops[0].__class__.__name__ == "Gt":
            code = "({} > {})".format(
                traverse(node.left, args, copy.copy(forvars), declared),
                traverse(node.comparators[0], args, copy.copy(forvars), declared),
            )
        elif len(node.ops) == 1 and node.ops[0].__class__.__name__ == "GtE":
            code = "({} >= {})".format(
                traverse(node.left, args, copy.copy(forvars), declared),
                traverse(node.comparators[0], args, copy.copy(forvars), declared),
            )
        elif len(node.ops) == 1 and node.ops[0].__class__.__name__ == "LtE":
            code = "({} <= {})".format(
                traverse(node.left, args, copy.copy(forvars), declared),
                traverse(node.comparators[0], args, copy.copy(forvars), declared),
            )
        else:
            raise Exception(
                "Unhandled Compare node {}. Please inform the developers.".format(
                    node.ops[0]
                )
            )
    elif node.__class__.__name__ == "AugAssign" or node.__class__.__name__ == "Assign":
        if node.__class__.__name__ == "Assign":
            assert len(node.targets) == 1
            operator = "="
            targetnode = node.targets[0]
        else:
            if node.op.__class__.__name__ == "Add":
                operator = "+="
            elif node.op.__class__.__name__ == "RShift":
                operator = ">>="
            elif node.op.__class__.__name__ == "LShift":
                operator = "<<="
            else:
                raise Exception(
                    f"Unhandled AugAssign node {node.op.__class__.__name__}"
                )
            targetnode = node.target
        left = traverse(targetnode, args, copy.copy(forvars), declared)
        if "[" in left:
            left = left[: left.find("[")]
        code = ""
        if left not in args.keys() and ("*" + left) not in args.keys():
            flag = True
        else:
            flag = False
        if node.value.__class__.__name__ == "Name" and (node.value.id in forvars):
            code = ""
            if flag and (
                traverse(targetnode, args, copy.copy(forvars), declared) not in declared
            ):
                code += "auto "
                declared.append(
                    traverse(targetnode, args, copy.copy(forvars), declared)
                )
            thread_var = getthread_dim(forvars.index(node.value.id))
            code += "{} = {};\n".format(
                traverse(targetnode, args, copy.copy(forvars), declared), thread_var
            )
        else:
            if node.value.__class__.__name__ == "IfExp":
                if flag and (
                    traverse(targetnode, args, copy.copy(forvars), declared)
                    not in declared
                ):
                    code = "auto {} {} {};\n".format(
                        traverse(targetnode, args, copy.copy(forvars), declared),
                        operator,
                        traverse(node.value.orelse, args, copy.copy(forvars), declared),
                    )
                    declared.append(
                        traverse(targetnode, args, copy.copy(forvars), declared)
                    )
                code += "if ({0}) {{\n {1} {2} {3};\n }} else {{\n {1} {2} {4};\n }}\n".format(
                    traverse(node.value.test, args, copy.copy(forvars), declared),
                    traverse(targetnode, args, copy.copy(forvars), declared),
                    operator,
                    traverse(node.value.body, args, copy.copy(forvars), declared),
                    traverse(node.value.orelse, args, copy.copy(forvars), declared),
                )
            elif node.value.__class__.__name__ == "Compare":
                assert len(node.value.ops) == 1
                code = ""
                if flag and (
                    traverse(targetnode, args, copy.copy(forvars), declared)
                    not in declared
                ):
                    code += "auto "
                    declared.append(
                        traverse(targetnode, args, copy.copy(forvars), declared)
                    )
                if node.value.ops[0].__class__.__name__ == "Lt":
                    compop = "<"
                elif node.value.ops[0].__class__.__name__ == "Gt":
                    compop = ">"
                elif node.value.ops[0].__class__.__name__ == "LtE":
                    compop = "<="
                elif node.value.ops[0].__class__.__name__ == "GtE":
                    compop = ">="
                elif node.value.ops[0].__class__.__name__ == "NotEq":
                    compop = "!="
                elif node.value.ops[0].__class__.__name__ == "Eq":
                    compop = "=="
                else:
                    raise Exception(
                        "Unhandled Compare node {}. Please inform the developers.".format(
                            node.value.ops[0]
                        )
                    )
                code += "{} {} {} {} {};\n".format(
                    traverse(targetnode, args, copy.copy(forvars), declared),
                    operator,
                    traverse(node.value.left, args, copy.copy(forvars), declared),
                    compop,
                    traverse(
                        node.value.comparators[0],
                        args,
                        copy.copy(forvars),
                        declared,
                    ),
                )
            else:
                code = ""
                if flag and (
                    traverse(targetnode, args, copy.copy(forvars), declared)
                    not in declared
                ):
                    code += "auto "
                    declared.append(
                        traverse(targetnode, args, copy.copy(forvars), declared)
                    )
                code += "{} {} {};\n".format(
                    traverse(targetnode, args, copy.copy(forvars), declared),
                    operator,
                    traverse(node.value, args, copy.copy(forvars), declared),
                )
    else:
        raise Exception(f"Unhandled node {node.__class__.__name__}")
    return code


def getbody(pycode, args):
    code = ""
    tree = ast.parse(pycode).body[0]
    declared = []
    for node in tree.body:
        code += traverse(node, args, [], declared)
    return code


def getxthreads(pycode):
    tree = ast.parse(pycode).body[0]
    forargs = set()
    flag = False
    while flag is False and "body" in dir(tree):
        for node in tree.body:
            if node.__class__.__name__ == "For":
                forargs.add(traverse(node.iter.args[0]))
                flag = True
            elif node.__class__.__name__ == "While":
                assert node.test.__class__.__name__ == "Compare"
                assert len(node.test.ops) == 1
                assert node.test.ops[0].__class__.__name__ == "Lt"
                assert len(node.test.comparators) == 1
                forargs.add(traverse(node.test.comparators[0]))
                flag = True
        tree = tree.body[0]
    if len(forargs) == 0:
        return 1
    elif len(forargs) == 1:
        return next(iter(forargs))
    else:
        lenarg = "std::max("
        count = 0
        for arg in forargs:
            if count == 0:
                lenarg += arg
                count += 1
            else:
                lenarg += ", " + arg
        lenarg += ")"
        return lenarg


def getythreads(pycode):
    tree = ast.parse(pycode).body[0]
    forargs = set()
    for node in tree.body:
        if node.__class__.__name__ == "For":
            for subnode in node.body:
                if subnode.__class__.__name__ == "For":
                    forargs.add(traverse(node.iter.args[0]))
    assert len(forargs) == 0 or len(forargs) == 1
    if len(forargs) == 0:
        return "1"
    elif len(forargs) == 1:
        return next(iter(forargs))
    else:
        raise Exception("Only doubly nested for loops can be handled")


def getctype(typename):
    pointercount = 0
    if "Const[" in typename:
        typename = typename[:-1]
        typename = typename.replace("Const[", "", 1)
        cpptype = "const "
    else:
        cpptype = ""
    while "List[" in typename:
        typename = typename[5:]
        typename = typename[:-1]
        pointercount += 1
    cpptype += typename + "*" * pointercount
    return cpptype


def gettemplateargs(spec):
    templateargs = OrderedDict()
    typelist = []
    templascii = 65
    for arg in spec["specializations"][0]["args"]:
        typelist.append(arg["type"])
    for i in range(len(spec["specializations"][0]["args"])):
        for childfunc in spec["specializations"]:
            if (
                typelist[i] != childfunc["args"][i]["type"]
                and childfunc["args"][i]["name"] not in templateargs.keys()
            ):
                templateargs[childfunc["args"][i]["name"]] = chr(templascii)
                templascii += 1
    return templateargs


def getparentargs(templateargs, spec):
    args = OrderedDict()
    for arg in spec["specializations"][0]["args"]:
        argname = arg["name"]
        if arg["name"] in templateargs.keys():
            if "*" in getctype(arg["type"]):
                argname = "*" + argname
            if "Const[" in arg["type"]:
                args[argname] = "const " + templateargs[arg["name"]]
            else:
                args[argname] = templateargs[arg["name"]]
        else:
            args[argname] = getctype(arg["type"])
    return args


def getchildargs(childfunc, spec):
    args = OrderedDict()
    for arg in childfunc["args"]:
        args[arg["name"]] = getctype(arg["type"])
    return args


def gettemplatestring(templateargs):
    count = 0
    templatestring = ""
    for x in templateargs.values():
        if count == 0:
            templatestring += "typename " + x
            count += 1
        else:
            templatestring += ", typename " + x
    return templatestring


def getdecl(name, args, templatestring, parent=False):
    code = ""
    if templatestring != "":
        code += "template <" + templatestring + ">\n"
    if parent:
        code += "__global__\n"
    count = 0
    for key, value in args.items():
        if count == 0:
            params = value + " " + key
            count += 1
        else:
            params += ", " + value + " " + key
    if parent:
        code += (
            "void "
            + name
            + "("
            + params
            + ", uint64_t invocation_index, uint64_t* err_code) {\n"
        )
    else:
        code += "ERROR " + name + "(" + params + ") {\n"
    return code


def gettemplatetypes(spec, templateargs):
    count = 0
    code = ""
    for arg in spec["args"]:
        for argname, info in arg.items():
            if argname in templateargs.keys():
                if count == 0:
                    code += getctype(info["type"]).replace("*", "")
                    count += 1
                else:
                    code += ", " + getctype(info["type"]).replace("*", "")
    if "const " in code:
        code = code.replace("const ", "")
    return code


def getparamnames(args):
    count = 0
    for arg in args.keys():
        if count == 0:
            code = arg
            count += 1
        else:
            code += ", " + arg
    return code


def getcode(indspec):
    templateargs = gettemplateargs(indspec)
    args = getparentargs(templateargs, indspec)
    templatestring = gettemplatestring(templateargs)
    code = getdecl(
        indspec["name"],
        args,
        templatestring,
        parent=True,
    )
    code += """  if (err_code[0] == NO_ERROR) {

    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
"""
    code += getbody(indspec["definition"], args)
    code += "}\n}\n"

    return code


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("kernelname", nargs="?")
    args = arg_parser.parse_args()
    kernelname = args.kernelname

    code = """// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

"""

    with open(os.path.join(CURRENT_DIR, "..", "kernel-specification.yml")) as specfile:
        indspec = yaml.safe_load(specfile)["kernels"]
        for spec in indspec:
            if spec["name"] == kernelname and (kernelname in KERNEL_WHITELIST):
                code = getcode(spec)
                print(code)
                break
            if kernelname is None and spec["name"] in KERNEL_WHITELIST:
                with open(
                    os.path.join(
                        os.path.dirname(CURRENT_DIR),
                        "src",
                        "awkward",
                        "_v2",
                        "_connect",
                        "cuda",
                        "cuda_kernels",
                        spec["name"] + ".cu",
                    ),
                    "w",
                ) as outfile:
                    outfile.write(code + getcode(spec))
