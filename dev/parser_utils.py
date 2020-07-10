# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import re
from collections import OrderedDict
from collections.abc import Iterable


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
            if (
                func is True
                and re.search("u?int\d{1,2}_t\*?", line) is not None
                and "=" not in line
                and (line.count(",") == 1 or ") {" in line)
            ):
                tokens[funcname][
                    line[re.search("u?int\d{1,2}_t\*?", line).span()[1] :]
                    .strip()
                    .replace(",", "")
                    .replace(") {", "")
                ] = re.search("u?int\d{1,2}_t", line).group()
            if func is True:
                while re.search("u?int\d{1,2}_t\*?", line) is not None:
                    line = line.replace(re.search("u?int\d{1,2}_t", line).group(), "int")
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
                            line = line.replace(x, "int")
            if func is True and line.find("bool") != -1:
                if line.find("bool*") != -1:
                    typename = "bool*"
                else:
                    typename = "bool"
                if "=" not in line and "(" not in line:
                    varname = line[line.find(typename) + len(typename) + 1 :]
                    varname = re.sub("[\W_]+", "", varname)
                    tokens[funcname][varname] = "bool"
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


def check_fail_func(filename):
    pfile, _ = preprocess(filename)
    func = False
    fail = []
    for line in pfile.splitlines():
        if func is False and re.search("\s.*\(", line):
            funcname = re.search("\s.*\(", line).group()[1:-1]
            func = True
            parans = []
            if line.count("{") > 0:
                for _ in range(line.count("{")):
                    parans.append("{")
        if func and line.count("{") > 0:
            for _ in range(line.count("{")):
                parans.append("{")
        if func and "return failure" in line:
            fail.append(funcname)
        if func and line.count("}") > 0:
            for _ in range(line.count("}")):
                parans.pop()
            if len(parans) == 0:
                func = False
    return fail


def parseheader(filename):
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
        if re.match("(in|out)param", check) is None:
            raise AssertionError(
                "Only inparam or outparam allowed. Not {0}".format(check)
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
                    if (key != "role") and (key != "instance"):
                        raise AssertionError("Only role and instance allowed")
                else:
                    val = line[i]
                    d[key] = val
        return {name: d}

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
