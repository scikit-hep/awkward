# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import os
import re
from collections import OrderedDict
from collections.abc import Iterable

import yaml

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def indent_code(code, indent):
    finalcode = ""
    for line in code.splitlines():
        finalcode += " " * indent + line + "\n"
    return finalcode


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


def genpykernels():
    prefix = """
import copy

kMaxInt64  = 9223372036854775806
kSliceNone = kMaxInt64 + 1

"""
    with open(os.path.join(CURRENT_DIR, "spec.yaml")) as infile:
        spec = yaml.safe_load(infile)["Kernels"]
        with open(os.path.join(CURRENT_DIR, "kernels.py"), "w") as outfile:
            outfile.write(prefix)
            for func in spec:
                outfile.write(func["specification"])
                if "specializations" in func.keys():
                    for childfunc in func["specializations"]:
                        outfile.write(childfunc["name"] + " = " + func["name"] + "\n")
                    outfile.write("\n\n")


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


def gettokens(ctokens, htokens):
    tokens = OrderedDict()
    for x in htokens.keys():
        tokens[x] = OrderedDict()
        for y in htokens[x].keys():
            tokens[x][y] = OrderedDict()
            for z, val in htokens[x][y].items():
                tokens[x][y][z] = val
            for i in ctokens[x]["args"]:
                if i["name"] == y:
                    if i["list"] > 0:
                        tokens[x][y]["array"] = i["list"]
                    else:
                        tokens[x][y]["array"] = 0
                    tokens[x][y]["type"] = i["type"]
    return tokens
