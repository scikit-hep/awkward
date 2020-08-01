# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import argparse
import copy
import importlib
import json
import os
import re
from collections import OrderedDict
from collections.abc import Iterable
from itertools import product

import pycparser
from lark import Lark

from parser_utils import indent_code

try:
    import black

    blackmode = black.FileMode()  # Initialize black config
    blackimported = True
except ImportError:
    blackimported = False


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

SPEC_BLACKLIST = [
    "awkward_ListArray_combinations_step",
    "awkward_Index8_getitem_at_nowrap",
    "awkward_IndexU8_getitem_at_nowrap",
    "awkward_Index32_getitem_at_nowrap",
    "awkward_IndexU32_getitem_at_nowrap",
    "awkward_Index64_getitem_at_nowrap",
    "awkward_NumpyArraybool_getitem_at",
    "awkward_NumpyArray8_getitem_at",
    "awkward_NumpyArrayU8_getitem_at",
    "awkward_NumpyArray16_getitem_at",
    "awkward_NumpyArray32_getitem_at",
    "awkward_NumpyArrayU32_getitem_at",
    "awkward_NumpyArray64_getitem_at",
    "awkward_NumpyArrayU64_getitem_at",
    "awkward_NumpyArrayfloat32_getitem_at",
    "awkward_NumpyArrayfloat64_getitem_at",
    "awkward_Index8_setitem_at_nowrap",
    "awkward_IndexU8_setitem_at_nowrap",
    "awkward_Index32_setitem_at_nowrap",
    "awkward_IndexU32_setitem_at_nowrap",
    "awkward_Index64_setitem_at_nowrap",
    "awkward_regularize_rangeslice",
    "awkward_NumpyArrayU16_getitem_at",
    "awkward_ListArray_combinations",
    "awkward_RegularArray_combinations",
    "awkward_ListOffsetArray_reduce_nonlocal_preparenext_64",
]

TEST_BLACKLIST = SPEC_BLACKLIST + [
    "awkward_ListArray_combinations",
    "awkward_RegularArray_combinations",
    "awkward_slicearray_ravel",
    "awkward_ListArray_getitem_next_range_carrylength",
    "awkward_ListArray_getitem_next_range",
    "awkward_regularize_rangeslice",
    "awkward_ListOffsetArray_rpad_and_clip_axis1",
    "awkward_ListOffsetArray_rpad_axis1",
    "awkward_UnionArray_flatten_combine",
    "awkward_UnionArray_flatten_length",
    "awkward_ListArray_getitem_jagged_numvalid",
    "awkward_NumpyArray_fill",
]

SUCCESS_TEST_BLACKLIST = TEST_BLACKLIST + [
    "awkward_RegularArray_broadcast_tooffsets",
    "awkward_ListArray_validity",
    "awkward_IndexedArray_validity",
    "awkward_UnionArray_validity",
    "awkward_regularize_arrayslice",
]

PYGEN_BLACKLIST = SPEC_BLACKLIST + [
    "awkward_sorting_ranges",
    "awkward_sorting_ranges_length",
    "awkward_argsort",
    "awkward_argsort_bool",
    "awkward_argsort_int8",
    "awkward_argsort_uint8",
    "awkward_argsort_int16",
    "awkward_argsort_uint16",
    "awkward_argsort_int32",
    "awkward_argsort_uint32",
    "awkward_argsort_int64",
    "awkward_argsort_uint64",
    "awkward_argsort_float32",
    "awkward_argsort_float64",
    "awkward_sort",
    "awkward_sort_bool",
    "awkward_sort_int8",
    "awkward_sort_uint8",
    "awkward_sort_int16",
    "awkward_sort_uint16",
    "awkward_sort_int32",
    "awkward_sort_uint32",
    "awkward_sort_int64",
    "awkward_sort_uint64",
    "awkward_sort_float32",
    "awkward_sort_float64",
    "awkward_ListOffsetArray_local_preparenext_64",
    "awkward_IndexedArray_local_preparenext_64",
    "awkward_NumpyArray_sort_asstrings_uint8",
    "awkward_NumpyArray_contiguous_copy",
    "awkward_NumpyArray_contiguous_copy_64",
    "awkward_NumpyArray_getitem_next_null",
    "awkward_NumpyArray_getitem_next_null_64",
]


def pytype(cpptype):
    cpptype = cpptype.replace("*", "")
    if re.match("u?int\d{1,2}(_t)?", cpptype) is not None:
        return "int"
    elif cpptype == "double":
        return "float"
    else:
        return cpptype


def getdirname(filename):
    if "/" in filename:
        name = filename[filename.rfind("/") + 1 : -4]
    else:
        name = filename[:-4]
    return name


def getheadername(filename):
    if "/" in filename:
        hfile = filename[filename.rfind("/") + 1 : -4] + ".h"
    else:
        hfile = filename[:-4] + ".h"
    hfile = os.path.join(CURRENT_DIR, "..", "include", "awkward", "cpu-kernels", hfile)
    return hfile


def arrayconv(cpptype):
    count = cpptype.count("*")
    if count == 0:
        return cpptype
    else:
        return "List[" * count + cpptype[:-count] + "]" * count


def preprocess(filename, skip_implementation=False):
    code = ""
    func = False
    templ = False
    tokens = OrderedDict()
    templateids = []
    templatecall = False
    tempids = []
    funcer = False
    with open(filename, "r") as f:
        for line in f:
            if line.endswith("\n"):
                line = line[:-1].rstrip() + "\n"
            else:
                line = line.rstrip()
            if line.startswith("#"):
                continue
            if re.search("//.*\n", line):
                line = re.sub("//.*\n", "\n", line)
            if line.startswith("template") and func is False:
                templ = True
            if "typename" in line:
                iterate = True
                while iterate:
                    if re.search("typename [^,]*,", line) is not None:
                        tempids.append(
                            line[
                                re.search("typename [^,]*,", line).span()[0]
                                + 9 : re.search("typename [^,]*,", line).span()[1]
                                - 1
                            ]
                        )
                        line = line[re.search("typename [^,]*,", line).span()[1] :]
                    if re.search("typename [^,]*,", line) is None:
                        iterate = False
                if re.search("typename [^,]*>", line) is not None:
                    tempids.append(
                        line[
                            re.search("typename [^,]*>", line).span()[0]
                            + 9 : re.search("typename [^,]*>", line).span()[1]
                            - 1
                        ]
                    )
                for x in tempids:
                    templateids.append(x + "*")
                for x in tempids:
                    templateids.append(x)
                continue
            if func is True and line.count("{") > 0 and not skip_implementation:
                for _ in range(line.count("{")):
                    parans.append("{")
            if func is False and re.search("\s.*\(", line):
                if skip_implementation and "{" not in line:
                    funcer = True
                funcname = re.search("\s.*\(", line).group()[1:-1]
                tokens[funcname] = OrderedDict()
                line = line.replace(line.split(" ")[0], "int", 1)
                func = True
                parans = []
                code += line
                if line.count("{") > 0:
                    for _ in range(line.count("{")):
                        parans.append("{")
                continue
            if func is True and "return awkward" in line:
                if re.search("return .*<", line) is not None:
                    x = line[
                        re.search("return .*<", line).span()[0]
                        + 6 : re.search("return .*<", line).span()[1]
                        - 1
                    ].strip()
                else:
                    x = line[
                        re.search("return .*\(", line).span()[0]
                        + 6 : re.search("return .*\(", line).span()[1]
                        - 1
                    ].strip()
                if "childfunc" not in tokens[x].keys():
                    tokens[x]["childfunc"] = set()
                tokens[x]["childfunc"].add(funcname)
                tokens[funcname]["gen"] = False
            if (
                func is True
                and re.search("<.*>", line) is not None
                and "||" not in line
                and "&&" not in line
            ):
                line = line.replace(re.search("<.*>", line).group(), "")
            elif (
                func is True
                and re.search("<.*\n", line) is not None
                and ";" not in line
                and ")" not in line
                and "[" not in line
                and "||" not in line
                and "&&" not in line
            ):
                templatecall = True
                line = line.replace(re.search("<.*\n", line).group(), "")
            elif (
                func is True
                and re.search(".*>", line) is not None
                and ";" not in line
                and "(" not in line[: re.search(".*>", line).span()[1]]
                and "||" not in line
                and "&&" not in line
            ):
                line = line.replace(re.search(".*>", line).group(), "")
                templatecall = False
            elif func is True and templatecall is True:
                line = ""
            if func is True:
                while re.search("u?int\d{1,2}_t\*?", line) is not None:
                    line = line.replace(
                        re.search("u?int\d{1,2}_t", line).group(), "int"
                    )
            if func is True and " ERROR " in line:
                line = line.replace("ERROR", "int", 1)
            if func is True and "(size_t)" in line:
                line = line.replace("(size_t)", "")
            if func is True and "std::" in line:
                line = line.replace("std::", "")
            if func is True and templ is True:
                for x in templateids:
                    if x in line:
                        if (
                            line[line.find(x) - 1] == " "
                            or line[line.find(x) - 1] == "*"
                            or line[line.find(x) - 1]
                        ):
                            if x.endswith("*"):
                                x = x[:-1]
                            if ("(" + x + ")") in line:
                                line = line.replace(x, "float")
                            else:
                                line = line.replace(x, "int")
            if func is True and line.find("bool") != -1:
                if line.find("bool*") != -1:
                    typename = "bool*"
                else:
                    typename = "bool"
                if "=" not in line and "(" not in line:
                    varname = line[line.find(typename) + len(typename) + 1 :]
                    varname = re.sub("[\W_]+", "", varname)
                line = line.replace("bool", "int", 1)
            if funcer and "{" in line and skip_implementation:
                funcer = False
            elif skip_implementation and "return" in line and "(" in line:
                if ")" not in line:
                    line = line.replace("\n", "")
                    if line.strip().endswith(";"):
                        line = line[:-1] + ")" + ";"
                    else:
                        line = line + ")" + ";"
                line = line + "\n" + "}" + "\n"
            elif skip_implementation and not funcer:
                continue
            if func and line.count("}") > 0:
                if not skip_implementation:
                    for _ in range(line.count("}")):
                        parans.pop()
                if len(parans) == 0:
                    func = False
                    templ = False
                    templateids = []
                    tempids = []
            code += line

    return code, tokens


class FuncBody(object):
    def __init__(self, ast):
        self.ast = ast
        self.code = ""
        self.traverse(self.ast.block_items, 4)

    def traverse(self, item, indent, called=False):
        if item.__class__.__name__ == "list":
            for node in item:
                self.traverse(node, indent)
        elif item.__class__.__name__ == "Return":
            if (
                item.expr.__class__.__name__ == "FuncCall"
                and item.expr.name.name == "failure"
            ):
                stmt = " " * indent + "raise ValueError({0})".format(
                    item.expr.args.exprs[0].value
                )
            elif (
                item.expr.__class__.__name__ == "FuncCall"
                and item.expr.name.name == "success"
                and item.expr.args is None
            ):
                stmt = " " * indent + "return"
            else:
                stmt = " " * indent + "return {0}".format(
                    self.traverse(item.expr, 0, called=True)
                )
            if called:
                return stmt
            else:
                self.code += stmt + "\n"
        elif item.__class__.__name__ == "Constant":
            stmt = " " * indent + "{0}".format(item.value)
            if called:
                return stmt
            else:
                self.code += stmt
        elif item.__class__.__name__ == "Decl":
            if item.init is not None:
                stmt = " " * indent + "{0} = {1}".format(
                    item.name, self.traverse(item.init, 0, called=True)
                )
                if not called:
                    stmt = stmt + "\n"
            elif item.type.__class__.__name__ == "PtrDecl":
                stmt = " " * indent + "{0} = []".format(item.name)
                if not called:
                    stmt = stmt + "\n"
            else:
                stmt = ""
            if called:
                return stmt
            else:
                self.code += stmt
        elif item.__class__.__name__ == "Assignment":
            if (
                item.rvalue.__class__.__name__ == "ArrayRef"
                and item.rvalue.subscript.__class__.__name__ == "BinaryOp"
                and item.rvalue.subscript.left.__class__.__name__ == "UnaryOp"
                and item.rvalue.subscript.left.op == "++"
            ):
                stmt = " " * indent + "{0} += 1; {1} = {2}[{0} {3} {4}]".format(
                    self.traverse(item.rvalue.subscript.left.expr, 0, called=True),
                    self.traverse(item.lvalue, 0, called=True),
                    self.traverse(item.rvalue.name, 0, called=True),
                    item.rvalue.subscript.op,
                    self.traverse(item.rvalue.subscript.right, 0, called=True),
                )
            else:
                stmt = " " * indent + "{0} {1} {2}".format(
                    self.traverse(item.lvalue, 0, called=True),
                    item.op,
                    self.traverse(item.rvalue, 0, called=True),
                )
            if called:
                return stmt
            else:
                self.code += stmt + "\n"
        elif item.__class__.__name__ == "FuncCall":
            if item.args is not None:
                if item.name.name == "memcpy":
                    return " " * indent + "{0}[{1}:{1}+{2}] = {3}[{4}:{4}+{2}]".format(
                        item.args.exprs[0].expr.name.name,
                        self.traverse(
                            item.args.exprs[0].expr.subscript, 0, called=True
                        ),
                        item.args.exprs[2].name,
                        item.args.exprs[1].expr.name.name,
                        self.traverse(
                            item.args.exprs[1].expr.subscript, 0, called=True
                        ),
                    )
                return " " * indent + "{0}({1})".format(
                    item.name.name, self.traverse(item.args, 0, called=True)
                )
            else:
                return " " * indent + "{0}()".format(item.name.name)
        elif item.__class__.__name__ == "ExprList":
            stmt = " " * indent
            for i in range(len(item.exprs)):
                if i == 0:
                    stmt += "{0}".format(self.traverse(item.exprs[i], 0, called=True))
                else:
                    stmt += ", {0}".format(self.traverse(item.exprs[i], 0, called=True))
            if called:
                return stmt
            else:
                self.code += stmt
        elif item.__class__.__name__ == "BinaryOp":
            if item.op == "&&":
                operator = "and"
            elif item.op == "||":
                operator = "or"
            else:
                operator = item.op
            binaryopl = "{0}".format(self.traverse(item.left, 0, called=True))
            binaryopr = "{0}".format(self.traverse(item.right, 0, called=True))
            if called and item.left.__class__.__name__ == "BinaryOp":
                binaryopl = "(" + binaryopl + ")"
            if called and item.right.__class__.__name__ == "BinaryOp":
                binaryopr = "(" + binaryopr + ")"
            stmt = " " * indent + "{0} {1} {2}".format(binaryopl, operator, binaryopr)
            if called:
                return stmt
            else:
                self.code += stmt
        elif item.__class__.__name__ == "If":
            stmt = " " * indent + "if {0}:\n".format(
                self.traverse(item.cond, 0, called=True)
            )
            stmt += "{0}\n".format(self.traverse(item.iftrue, indent + 4, called=True))
            if item.iffalse is not None:
                stmt += " " * indent + "else:\n"
                stmt += "{0}\n".format(
                    self.traverse(item.iffalse, indent + 4, called=True)
                )
            if called:
                return stmt[:-1]
            else:
                self.code += stmt
        elif item.__class__.__name__ == "For":
            if (
                (item.init is not None)
                and (item.next is not None)
                and (item.cond is not None)
                and (len(item.init.decls) == 1)
                and (
                    item.init.decls[0].init.__class__.__name__ == "Constant"
                    or item.init.decls[0].init.__class__.__name__ == "ID"
                )
                and (item.next.op == "p++")
                and (item.cond.op == "<")
                and (item.cond.left.name == item.init.decls[0].name)
            ):
                if item.init.decls[0].init.__class__.__name__ == "Constant":
                    if item.init.decls[0].init.value == "0":
                        stmt = " " * indent + "for {0} in range({1}):\n".format(
                            item.init.decls[0].name,
                            self.traverse(item.cond.right, 0, called=True),
                        )
                    else:
                        stmt = " " * indent + "for {0} in range({1}, {2}):\n".format(
                            item.init.decls[0].name,
                            item.init.decls[0].init.value,
                            self.traverse(item.cond.right, 0, called=True),
                        )
                else:
                    stmt = " " * indent + "for {0} in range({1}, {2}):\n".format(
                        item.init.decls[0].name,
                        item.init.decls[0].init.name,
                        self.traverse(item.cond.right, 0, called=True),
                    )
                for i in range(len(item.stmt.block_items)):
                    stmt += (
                        self.traverse(item.stmt.block_items[i], indent + 4, called=True)
                        + "\n"
                    )
            else:
                if item.init is not None:
                    stmt = "{0}\n".format(self.traverse(item.init, indent, called=True))
                else:
                    stmt = ""
                stmt += " " * indent + "while {0}:\n".format(
                    self.traverse(item.cond, 0, called=True)
                )
                for i in range(len(item.stmt.block_items)):
                    stmt += (
                        self.traverse(item.stmt.block_items[i], indent + 4, called=True)
                        + "\n"
                    )
                stmt += " " * (indent + 4) + "{0}\n".format(
                    self.traverse(item.next, 0, called=True)
                )
            if called:
                return stmt[:-1]
            else:
                self.code += stmt
        elif item.__class__.__name__ == "UnaryOp":
            if item.op[1:] == "++" or item.op == "++":
                stmt = " " * indent + "{0} = {0} + 1".format(
                    self.traverse(item.expr, 0, called=True)
                )
            elif item.op[1:] == "--":
                stmt = " " * indent + "{0} = {0} - 1".format(
                    self.traverse(item.expr, 0, called=True)
                )
            elif item.op == "*":
                stmt = " " * indent + "{0}[0]".format(
                    self.traverse(item.expr, 0, called=True)
                )
            elif item.op == "-":
                stmt = " " * indent + "-{0}".format(
                    self.traverse(item.expr, 0, called=True)
                )
            elif item.op == "!":
                stmt = " " * indent + "not ({0})".format(
                    self.traverse(item.expr, 0, called=True)
                )
            elif item.op == "&":
                stmt = " " * indent + "{0}".format(
                    self.traverse(item.expr, 0, called=True)
                )
            else:
                raise NotImplementedError(
                    "Unhandled Unary Operator case. Please inform the developers about the error"
                )
            if called:
                return stmt
            else:
                self.code += stmt + "\n"
        elif item.__class__.__name__ == "DeclList":
            stmt = " " * indent
            for i in range(len(item.decls)):
                if i == 0:
                    stmt += "{0}".format(self.traverse(item.decls[i], 0, called=True))
                else:
                    stmt += ", {0}".format(self.traverse(item.decls[i], 0, called=True))
            if called:
                return stmt
            else:
                self.code += stmt + "\n"
        elif item.__class__.__name__ == "ArrayRef":
            if (
                item.subscript.__class__.__name__ == "UnaryOp"
                and item.subscript.op[1:] == "++"
            ):
                stmt = (
                    " " * indent
                    + "{0};".format(self.traverse(item.subscript, 0, called=True))
                    + " {0}[{1}]".format(
                        self.traverse(item.name, 0, called=True),
                        self.traverse(item.subscript.expr, 0, called=True),
                    )
                )
            else:
                stmt = " " * indent + "{0}[{1}]".format(
                    self.traverse(item.name, 0, called=True),
                    self.traverse(item.subscript, 0, called=True),
                )
            if called:
                return stmt
            else:
                self.code += stmt
        elif item.__class__.__name__ == "Cast":
            stmt = " " * indent + "{0}({1})".format(
                self.traverse(item.to_type, 0, called=True),
                self.traverse(item.expr, 0, called=True),
            )
            if called:
                return stmt
            else:
                self.code += stmt
        elif item.__class__.__name__ == "Typename":
            stmt = " " * indent + "{0}".format(item.type.type.names[0])
            if called:
                return stmt
            else:
                self.code += stmt
        elif item.__class__.__name__ == "ID":
            if item.name == "true":
                name = "True"
            elif item.name == "false":
                name = "False"
            else:
                name = item.name
            stmt = " " * indent + "{0}".format(name)
            if called:
                return stmt
            else:
                self.code += stmt
        elif item.__class__.__name__ == "Compound":
            stmt = ""
            if called:
                for i in range(len(item.block_items)):
                    stmt += (
                        self.traverse(item.block_items[i], indent, called=True) + "\n"
                    )
            else:
                for i in range(len(item.block_items)):
                    stmt += (
                        self.traverse(item.block_items[i], indent + 4, called=True)
                        + "\n"
                    )
            return stmt[:-1]
        elif item.__class__.__name__ == "TernaryOp":
            stmt = " " * indent + "{0} if {1} else {2}".format(
                self.traverse(item.iftrue, 0, called=True),
                self.traverse(item.cond, 0, called=True),
                self.traverse(item.iffalse, 0, called=True),
            )
            if called:
                return stmt
            else:
                self.code += stmt
        elif item.__class__.__name__ == "While":
            stmt = " " * indent + "while {0}:\n".format(
                self.traverse(item.cond, 0, called=True)
            )
            for i in range(len(item.stmt.block_items)):
                stmt += (
                    self.traverse(item.stmt.block_items[i], indent + 4, called=True)
                    + "\n"
                )
            if called:
                return stmt
            else:
                self.code += stmt
        elif item.__class__.__name__ == "StructRef":
            stmt = " " * indent + "{0}{1}{2}".format(
                self.traverse(item.name, 0, called=True),
                item.type,
                self.traverse(item.field, 0, called=True),
            )
            if called:
                return stmt
            else:
                self.code += stmt
        elif item.__class__.__name__ == "EmptyStatement":
            pass
        else:
            raise Exception("Unable to parse {0}".format(item.__class__.__name__))


class FuncDecl(object):
    def __init__(self, ast):
        self.ast = ast
        self.name = ast.name
        self.args = []
        self.returntype = self.ast.type.type.type.names[0]
        self.traverse()

    def traverse(self):
        if self.ast.type.args is not None:
            params = self.ast.type.args.params
            for param in params:
                typename, listcount = self.iterclass(param.type, 0)
                self.args.append(
                    {"name": param.name, "type": typename, "list": listcount}
                )

    def iterclass(self, obj, count):
        if obj.__class__.__name__ == "IdentifierType":
            return obj.names[0], count
        elif obj.__class__.__name__ == "TypeDecl":
            return self.iterclass(obj.type, count)
        elif obj.__class__.__name__ == "PtrDecl":
            return self.iterclass(obj.type, count + 1)

    def arrange_args(self):
        arranged = ""
        for i in range(len(self.args)):
            if i != 0:
                arranged += ", "
            arranged += "{0} ".format(self.args[i]["name"])
        return arranged


def remove_return(code):
    if code[code.rfind("\n", 0, code.rfind("\n")) :].strip() == "return":
        k = code.rfind("return")
        code = code[:k] + code[k + 6 :]
    return code


def genpython(pfile):
    ast = pycparser.c_parser.CParser().parse(pfile)
    funcs = OrderedDict()
    for i in range(len(ast.ext)):
        decl = FuncDecl(ast.ext[i].decl)
        if decl.name not in SPEC_BLACKLIST:
            if blackimported:
                funcs[decl.name] = black.format_str(
                    (
                        "def {0}({1})".format(decl.name, decl.arrange_args(),)
                        + ":\n"
                        + remove_return(FuncBody(ast.ext[i].body).code)
                    ),
                    mode=blackmode,
                )
            else:
                funcs[decl.name] = (
                    "def {0}({1})".format(decl.name, decl.arrange_args(),)
                    + ":\n"
                    + remove_return(FuncBody(ast.ext[i].body).code)
                )
    return funcs


def getargs(filename):
    def traverse(tree, funcdict):
        if isinstance(tree, list):
            for node in tree:
                traverse(node, funcdict)
        elif tree.data == "file":
            traverse(tree.children, funcdict)
        elif tree.data == "def":
            if tree.children[0] == "struct Error":
                funcdict[tree.children[1]] = OrderedDict()
                assert tree.children[2].data == "args"
                for arg in tree.children[2].children:
                    assert arg.data == "pair"
                    funcdict[tree.children[1]][arg.children[1]] = arg.children[0]

    pydef_parser = Lark(
        r"""
    file: "extern" "\"C\"" "{" def* "}"
    def: "EXPORT_SYMBOL" RET FUNCNAME "(" args ");"

    FUNCNAME: CNAME
    pair: TYPE PARAMNAME
    args: pair ("," pair)*
    TYPE: /u?int\d{1,2}_t\*?\*?/
        | /bool\*?/
        | /float\*?/
        | /double\*?/
    PARAMNAME: CNAME
    RET: "struct Error"
       | "void"
       | /u?int\d{1,2}_t/
       | "bool"
       | "float"
       | "double"
    DONTREAD: /\/\/[^\n]*/
            | /#ifndef[^\n]*/
            | /#define[^\n]*/
            | /#include[^\n]*/
            | /#endif[^\n]*/
            | "const"

    %import common.CNAME
    %import common.WS
    %ignore WS
    %ignore DONTREAD
    """,
        start="file",
    )
    funcs = OrderedDict()
    with open(filename) as f:
        fstr = f.read()
        traverse(pydef_parser.parse(fstr), funcs)
    return funcs


def commentparser(line):
    def flatten(l):
        for el in l:
            if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                yield from flatten(el)
            else:
                yield el

    d = OrderedDict()
    line = line[line.find("@param") + len("@param") + 1 :]
    line = re.sub(" +", " ", line)
    name = line.split()[0]
    check = line.split()[1]
    if re.match("(in|out|inout)param", check) is None:
        raise AssertionError(
            "Only inparam. outparam or inoutparam allowed. Not {0}".format(check)
        )
    d["check"] = check
    line = line[line.find(check) + len(check) + 1 :]
    if line.strip() != "":
        line = [x.strip() for x in line.split(":")]
        for i in range(len(line)):
            line[i] = line[i].split(" ")
        line = list(flatten(line))
        if len(line) % 2 != 0:
            raise AssertionError("Un-paired labels not allowed")
        for i in range(len(line)):
            if i % 2 == 0:
                key = line[i]
                if key != "role":
                    raise AssertionError("Only role allowed")
            else:
                val = line[i]
                d[key] = val
    return {name: d}


def parseheader(filename):
    with open(filename, "r") as f:
        tokens = OrderedDict()
        funcs = OrderedDict()
        for line in f:
            if "///@param" in line.replace(" ", ""):
                tokens.update(commentparser(line))
                continue
            elif "awkward_" in line:
                funcname = line[line.find("awkward_") : line.find("(")].strip()
                funcs[funcname] = tokens
                tokens = OrderedDict()
            else:
                continue
        return funcs


def get_tests(allpykernels, allfuncargs, alltokens, allfuncroles):
    def writepykernels():
        prefix = """

kMaxInt64  = 9223372036854775806
kSliceNone = kMaxInt64 + 1

"""
        with open(os.path.join(CURRENT_DIR, "kernels.py"), "w") as f:
            f.write(prefix)
            for func in alltokens.keys():
                if (
                    "gen" not in alltokens[func].keys()
                    and func not in TEST_BLACKLIST
                    and func not in PYGEN_BLACKLIST
                ):
                    f.write(allpykernels[func] + "\n")
                    if "childfunc" in alltokens[func].keys():
                        for childfunc in alltokens[func]["childfunc"]:
                            f.write(childfunc + " = " + func + "\n")
                        f.write("\n")

    writepykernels()
    import kernels

    importlib.reload(kernels)

    with open(
        os.path.join(CURRENT_DIR, "..", "kernel-specification", "samples.json")
    ) as testjson:
        data = json.load(testjson)
        funcs = OrderedDict()
        for funcname in alltokens.keys():
            if (
                "gen" not in alltokens[funcname].keys()
                and funcname not in TEST_BLACKLIST
                and funcname not in PYGEN_BLACKLIST
            ):
                if "childfunc" in alltokens[funcname].keys():
                    keyfunc = next(iter(alltokens[funcname]["childfunc"]))
                else:
                    keyfunc = funcname

                funcs[funcname] = []
                funcPy = getattr(kernels, funcname)

                firstdict = {}
                instancedict = {}
                checkindex = []

                count = 0
                for i in range(len(allfuncargs[keyfunc].keys())):
                    arg = list(allfuncargs[keyfunc].keys())[i]
                    argpytype = pytype(allfuncargs[keyfunc][arg])
                    firstdict[arg] = []
                    if allfuncroles[keyfunc][arg]["check"] == "inparam":
                        if "role" in allfuncroles[keyfunc][arg].keys():
                            group = allfuncroles[keyfunc][arg]["role"][
                                : allfuncroles[keyfunc][arg]["role"].find("-")
                            ]
                            if group not in instancedict.keys():
                                instancedict[group] = []
                            instancedict[group].append(arg)
                            for x in range(
                                len(data[group][allfuncroles[keyfunc][arg]["role"]])
                            ):
                                firstdict[arg].append(
                                    data[group][allfuncroles[keyfunc][arg]["role"]][x][
                                        argpytype
                                    ]
                                )
                        else:
                            group = str(count)
                            assert group not in instancedict.keys()
                            instancedict[group] = [arg]
                            firstdict[arg].append(data["num"])
                            assert len(firstdict[arg]) == 1
                            count += 1
                    elif allfuncroles[keyfunc][arg]["check"] == "outparam":
                        group = str(count)
                        assert group not in instancedict.keys()
                        instancedict[group] = [arg]
                        firstdict[arg].append({})
                        assert (len(firstdict[arg])) == 1
                        checkindex.append(i)
                        count += 1
                    else:
                        raise AssertionError(
                            "Function argument must have inparam/outparam role"
                        )

                instancedictlist = list(instancedict.keys())

                combinations = []
                for name in instancedictlist:
                    temp = []
                    for arg in instancedict[name]:
                        temp.append(firstdict[arg])
                    combinations.append(zip(*temp))

                for x in product(*combinations):
                    temp = {}
                    for groupName, t in zip(instancedictlist, x):
                        for key, value in zip(instancedict[groupName], t):
                            temp[key] = value

                    tests = []
                    intests = OrderedDict()
                    outtests = OrderedDict()
                    for key in firstdict.keys():
                        argpytype = pytype(allfuncargs[keyfunc][key])
                        if allfuncroles[keyfunc][key]["check"] == "outparam":
                            if argpytype == "int":
                                intests[key] = [123] * 50
                            elif argpytype == "float":
                                intests[key] = [123.0] * 50
                            elif argpytype == "bool":
                                intests[key] = [True] * 50
                        elif allfuncroles[keyfunc][key]["check"] == "inparam":
                            intests[key] = temp[key]
                        else:
                            raise AssertionError(
                                "Invalid roles should have been filtered"
                            )
                        tests.append(copy.deepcopy(temp[key]))

                    try:
                        funcPy(*tests)
                        if funcname not in SUCCESS_TEST_BLACKLIST:
                            for i in checkindex:
                                count = 0
                                outtests[list(allfuncargs[keyfunc].keys())[i]] = []
                                for num in sorted(tests[i]):
                                    if num != count:
                                        while num != count:
                                            outtests[
                                                list(allfuncargs[keyfunc].keys())[i]
                                            ].append(123)
                                            count = count + 1
                                    outtests[
                                        list(allfuncargs[keyfunc].keys())[i]
                                    ].append(tests[i][num])
                                    count = count + 1
                            tempdict = {}
                            tempdict["input"] = copy.deepcopy(intests)
                            tempdict["output"] = copy.copy(outtests)
                            tempdict["success"] = True
                            funcs[funcname].append(tempdict)
                    except ValueError:
                        tempdict = {}
                        tempdict["input"] = copy.deepcopy(intests)
                        tempdict["success"] = False
                        funcs[funcname].append(tempdict)

    return funcs


def get_paramcheck(funcroles, alltokens, allfuncargs):
    funcs = OrderedDict()
    for funcname in alltokens.keys():
        if "gen" not in alltokens[funcname].keys() and funcname not in SPEC_BLACKLIST:
            if "childfunc" in alltokens[funcname].keys():
                keyfunc = next(iter(alltokens[funcname]["childfunc"]))
            else:
                keyfunc = funcname
            funcs[funcname] = OrderedDict()
            for arg in allfuncargs[keyfunc].keys():
                funcs[funcname][arg] = funcroles[keyfunc][arg]["check"]
    return funcs


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("kernelname", nargs="?")
    args = arg_parser.parse_args()
    kernelname = args.kernelname

    kernelfiles = [
        os.path.join(CURRENT_DIR, "..", "src", "cpu-kernels", "identities.cpp",),
        os.path.join(CURRENT_DIR, "..", "src", "cpu-kernels", "operations.cpp",),
        os.path.join(CURRENT_DIR, "..", "src", "cpu-kernels", "reducers.cpp",),
        os.path.join(CURRENT_DIR, "..", "src", "cpu-kernels", "getitem.cpp",),
        os.path.join(CURRENT_DIR, "..", "src", "cpu-kernels", "sorting.cpp"),
    ]

    with open(
        os.path.join(CURRENT_DIR, "..", "kernel-specification", "kernelnames.yml"), "w"
    ) as mainspec:
        mainspec.write("kernels:\n")
        for filename in kernelfiles:
            mainspec.write(" " * 2 + getdirname(filename) + ":\n")
            if "sorting.cpp" in filename:
                pfile, tokens = preprocess(filename, skip_implementation=True)
            else:
                pfile, tokens = preprocess(filename)
                pyfuncs = genpython(pfile)

            hfile = getheadername(filename)
            funcroles = parseheader(hfile)
            funcargs = getargs(hfile)

            if "sorting.cpp" not in filename:
                tests = get_tests(pyfuncs, funcargs, tokens, funcroles)
            paramchecks = get_paramcheck(funcroles, tokens, funcargs)

            if kernelname is None:
                for funcname in tokens.keys():
                    if (
                        "gen" not in tokens[funcname].keys()
                        and funcname not in SPEC_BLACKLIST
                    ):
                        try:
                            os.mkdir(
                                os.path.join(
                                    CURRENT_DIR,
                                    "..",
                                    "kernel-specification",
                                    getdirname(filename),
                                )
                            )
                        except FileExistsError:
                            pass
                        with open(
                            os.path.join(
                                CURRENT_DIR,
                                "..",
                                "kernel-specification",
                                getdirname(filename),
                                funcname + ".yml",
                            ),
                            "w",
                        ) as f:
                            mainspec.write(
                                " " * 4
                                + funcname
                                + ": "
                                + os.path.join(
                                    getdirname(filename), funcname + ".yml\n",
                                )
                            )
                            f.write("- name: " + funcname + "\n")
                            if "childfunc" in tokens[funcname].keys():
                                f.write(" " * 2 + "specializations:\n")
                                for childfunc in tokens[funcname]["childfunc"]:
                                    f.write(" " * 4 + "- name: " + childfunc + "\n")
                                    f.write(" " * 6 + "args:\n")
                                    for arg in funcargs[childfunc].keys():
                                        f.write(
                                            " " * 8
                                            + "- "
                                            + arg
                                            + ": "
                                            + arrayconv(funcargs[childfunc][arg])
                                            + "\n"
                                        )
                            else:
                                f.write(" " * 2 + "args:\n")
                                for arg in funcargs[funcname].keys():
                                    f.write(
                                        " " * 4
                                        + "- "
                                        + arg
                                        + ": "
                                        + arrayconv(funcargs[funcname][arg])
                                        + "\n"
                                    )
                            inparams = []
                            outparams = []
                            for arg in paramchecks[funcname].keys():
                                if (
                                    paramchecks[funcname][arg] == "inparam"
                                    or paramchecks[funcname][arg] == "inoutparam"
                                ):
                                    inparams.append(str(arg))
                                if (
                                    paramchecks[funcname][arg] == "outparam"
                                    or paramchecks[funcname][arg] == "inoutparam"
                                ):
                                    outparams.append(str(arg))
                            f.write(" " * 2 + "inparams: " + str(inparams) + "\n")
                            f.write(" " * 2 + "outparams: " + str(outparams) + "\n")
                            f.write(" " * 2 + "definition: |\n")
                            if funcname in PYGEN_BLACKLIST or "sorting.cpp" in filename:
                                f.write(" " * 4 + "Insert Python definition here\n")
                            else:
                                f.write(
                                    indent_code(pyfuncs[funcname], 4).rstrip() + "\n"
                                )
                            if (
                                "sorting.cpp" not in filename
                                and funcname not in TEST_BLACKLIST
                                and funcname in tests.keys()
                            ):
                                f.write(" " * 2 + "tests:\n")
                                for test in tests[funcname]:
                                    f.write(" " * 4 + "- args:\n")
                                    for arg in test["input"].keys():
                                        f.write(
                                            " " * 8
                                            + arg
                                            + ": "
                                            + str(test["input"][arg])
                                            + "\n"
                                        )
                                    f.write(
                                        " " * 6
                                        + "successful: "
                                        + str(test["success"])
                                        + "\n"
                                    )
                                    if test["success"] is True:
                                        f.write(" " * 6 + "results:\n")
                                        for arg in test["output"].keys():
                                            f.write(
                                                " " * 8
                                                + arg
                                                + ": "
                                                + str(test["output"][arg])
                                                + "\n"
                                            )
            else:
                if kernelname in tokens.keys() and kernelname not in SPEC_BLACKLIST:
                    print("name: ", kernelname)
                    if "childfunc" in tokens[kernelname].keys():
                        print("specializations:")
                        for childfunc in tokens[kernelname]["childfunc"]:
                            print(" " * 2 + "- name: " + childfunc)
                            print(" " * 4 + "args:")
                            for arg in funcargs[childfunc].keys():
                                print(
                                    " " * 6
                                    + "- "
                                    + arg
                                    + ": "
                                    + arrayconv(funcargs[childfunc][arg])
                                )
                    else:
                        for arg in funcargs[kernelname].keys():
                            print(
                                " " * 2
                                + "- "
                                + arg
                                + ": "
                                + arrayconv(funcargs[kernelname][arg])
                            )
                    inparams = []
                    outparams = []
                    for arg in paramchecks[kernelname].keys():
                        if (
                            paramchecks[kernelname][arg] == "inparam"
                            or paramchecks[kernelname][arg] == "inoutparam"
                        ):
                            inparams.append(str(arg))
                        if (
                            paramchecks[kernelname][arg] == "outparam"
                            or paramchecks[kernelname][arg] == "inoutparam"
                        ):
                            outparams.append(str(arg))
                    print(" " * 4 + "inparams: ", inparams)
                    print(" " * 4 + "outparams: ", outparams)
                    print(" " * 4 + "definition: |")
                    if kernelname in PYGEN_BLACKLIST or "sorting.cpp" in filename:
                        print(" " * 6 + "Insert Python definition here")
                    else:
                        print(indent_code(pyfuncs[kernelname], 6).rstrip())
                    if kernelname in tests.keys():
                        print(" " * 4 + "tests:")
                        for test in tests[kernelname]:
                            print(" " * 6 + "- args:")
                            for arg in test["input"].keys():
                                print(" " * 10 + arg + ": ", test["input"][arg])
                            print(" " * 8 + "successful: ", test["success"])
                            print(" " * 8 + "results:")
                            if test["success"] is True:
                                for arg in test["output"].keys():
                                    print(" " * 10 + arg + ": ", test["output"][arg])
                        print()
                else:
                    raise ValueError("Function {0} not present".format(kernelname))
