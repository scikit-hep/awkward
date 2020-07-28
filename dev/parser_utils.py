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


def pytype(cpptype):
    cpptype = cpptype.replace("*", "")
    if re.match("u?int\d{1,2}(_t)?", cpptype) is not None:
        return "int"
    elif cpptype == "double":
        return "float"
    else:
        return cpptype


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
