# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import re
from collections import OrderedDict
from collections.abc import Iterable


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
                    if (
                        (key != "role")
                        and (key != "instance")
                        and (key != "upperlimit")
                    ):
                        raise AssertionError(
                            "Only role, instance and upperlimit allowed"
                        )
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
