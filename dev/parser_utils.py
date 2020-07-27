# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import os
import re
from collections import OrderedDict
from collections.abc import Iterable

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
