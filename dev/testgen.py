# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import os
import subprocess
from collections import OrderedDict

import yaml

from parser_utils import pytype

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def genpykernels():
    print("Generating Python kernels")
    prefix = """

kMaxInt64  = 9223372036854775806
kSliceNone = kMaxInt64 + 1

"""
    with open(os.path.join(CURRENT_DIR, "spec", "spec.yaml")) as infile:
        mainspec = yaml.safe_load(infile)["kernels"]
        with open(
            os.path.join(CURRENT_DIR, "..", "tests-kernels", "kernels.py"), "w"
        ) as outfile:
            outfile.write(prefix)
            for filedir in mainspec.values():
                for relpath in filedir.values():
                    with open(os.path.join(CURRENT_DIR, "spec", relpath)) as specfile:
                        indspec = yaml.safe_load(specfile)[0]
                        if "def " in indspec["definition"]:
                            outfile.write(indspec["definition"] + "\n")
                            if "specializations" in indspec.keys():
                                for childfunc in indspec["specializations"]:
                                    outfile.write(
                                        childfunc["name"]
                                        + " = "
                                        + indspec["name"]
                                        + "\n"
                                    )
                                outfile.write("\n\n")


def readspec():
    funcs = {}
    with open(os.path.join(CURRENT_DIR, "spec", "spec.yaml")) as infile:
        mainspec = yaml.safe_load(infile)["kernels"]
        for filedir in mainspec.values():
            for relpath in filedir.values():
                with open(os.path.join(CURRENT_DIR, "spec", relpath)) as specfile:
                    indspec = yaml.safe_load(specfile)[0]
                    if "tests" in indspec.keys():
                        if "specializations" in indspec.keys():
                            for childfunc in indspec["specializations"]:
                                funcs[childfunc["name"]] = []
                                if indspec["tests"] is None:
                                    raise AssertionError(
                                        "No tests in specification for {0}".format(
                                            indspec["name"]
                                        )
                                    )
                                else:
                                    for test in indspec["tests"]:
                                        # Check if test has correct types
                                        flag = True
                                        count = 0
                                        for arg, val in test["args"].items():
                                            spectype = pytype(
                                                childfunc["args"][count][arg]
                                                .replace("List", "")
                                                .replace("[", "")
                                                .replace("]", "")
                                            )
                                            while isinstance(val, list):
                                                val = val[0]
                                            if type(val) != eval(spectype):
                                                flag = False
                                            count += 1
                                        if flag:
                                            testinfo = {}
                                            testinfo["inargs"] = OrderedDict()
                                            testinfo["inargs"].update(test["args"])
                                            testinfo["success"] = test["successful"]
                                            if testinfo["success"]:
                                                testinfo["outargs"] = OrderedDict()
                                                testinfo["outargs"].update(
                                                    test["results"]
                                                )
                                            funcs[childfunc["name"]].append(testinfo)
                        else:
                            funcs[indspec["name"]] = []
                            if indspec["tests"] is None:
                                raise AssertionError(
                                    "No tests in specification for {0}".format(
                                        indspec["name"]
                                    )
                                )
                            else:
                                for test in indspec["tests"]:
                                    # Check if test has correct types
                                    flag = True
                                    count = 0
                                    for arg, val in test["args"].items():
                                        spectype = pytype(
                                            indspec["args"][count][arg]
                                            .replace("List", "")
                                            .replace("[", "")
                                            .replace("]", "")
                                        )
                                        while isinstance(val, list):
                                            val = val[0]
                                        if type(val) != eval(spectype):
                                            flag = False
                                        count += 1
                                    if flag:
                                        testinfo = {}
                                        testinfo["inargs"] = OrderedDict()
                                        testinfo["inargs"].update(test["args"])
                                        testinfo["success"] = test["successful"]
                                        if testinfo["success"]:
                                            testinfo["outargs"] = OrderedDict()
                                            testinfo["outargs"].update(test["results"])
                                        funcs[indspec["name"]].append(testinfo)
    return funcs


def testpykernels(tests):
    print("Generating file for testing python kernels")
    for funcname in tests.keys():
        with open(
            os.path.join(
                CURRENT_DIR, "..", "tests-kernels", "test_py" + funcname + ".py"
            ),
            "w",
        ) as f:
            f.write("import pytest\nimport kernels\n\n")
            num = 1
            for test in tests[funcname]:
                if test == []:
                    raise AssertionError(
                        "Put proper tests for {0} in specification".format(funcname)
                    )
                f.write("def test_py" + funcname + "_" + str(num) + "():\n")
                num += 1
                args = ""
                for arg, val in test["inargs"].items():
                    f.write(" " * 4 + arg + " = " + str(val) + "\n")
                f.write(" " * 4 + "funcPy = getattr(kernels, '" + funcname + "')\n")
                count = 0
                for arg in test["inargs"].keys():
                    if count == 0:
                        args += arg
                        count += 1
                    else:
                        args += ", " + arg
                if test["success"]:
                    f.write(" " * 4 + "funcPy" + "(" + args + ")\n")
                    for arg, val in test["outargs"].items():
                        f.write(" " * 4 + "out" + arg + " = " + str(val) + "\n")
                        if isinstance(val, list):
                            f.write(" " * 4 + "for i in range(len(out" + arg + ")):\n")
                            f.write(
                                " " * 8 + "assert " + arg + "[i] == out" + arg + "[i]\n"
                            )
                        else:
                            f.write(" " * 4 + "assert " + arg + " == out" + arg + "\n")
                else:
                    f.write(" " * 4 + "with pytest.raises(Exception):\n")
                    f.write(" " * 8 + "funcPy(" + args + ")\n")
                f.write("\n")


def testcpukernels(tests):
    print("Generating file for testing CPU kernels")

    def getctypelist(typedict):
        newctypes = []
        for typename in typedict.values():
            if "List" in typename:
                count = typename.count("List")
                typename = (
                    typename.replace("List", "").replace("[", "").replace("]", "")
                )
                if typename.endswith("_t"):
                    typename = typename[:-2]
                start = ""
                end = ")"
                for i in range(count):
                    if i > 0:
                        start += "("
                        end += ")"
                    start += "ctypes.POINTER"
                start += "(ctypes.c_"
                newctypes.append(start + typename + end)
            else:
                if typename.endswith("_t"):
                    typename = typename[:-2]
                newctypes.append("ctypes.c_" + typename)
        count = 0
        funcCtypes = "("
        for x in newctypes:
            if count == 0:
                funcCtypes += x
                count += 1
            else:
                funcCtypes += ", " + x
        funcCtypes += ")"
        return funcCtypes

    def getfuncargs():
        funcs = {}
        with open(os.path.join(CURRENT_DIR, "spec", "spec.yaml")) as infile:
            mainspec = yaml.safe_load(infile)["kernels"]
            for filedir in mainspec.values():
                for relpath in filedir.values():
                    with open(os.path.join(CURRENT_DIR, "spec", relpath)) as specfile:
                        indspec = yaml.safe_load(specfile)[0]
                        if (
                            "def " in indspec["definition"]
                            and "tests" in indspec.keys()
                            and indspec["tests"] is not None
                        ):
                            if "specializations" in indspec.keys():
                                for childfunc in indspec["specializations"]:
                                    funcs[childfunc["name"]] = {}
                                    for arg in childfunc["args"]:
                                        funcs[childfunc["name"]][
                                            list(arg.keys())[0]
                                        ] = list(arg.values())[0]
                            else:
                                funcs[indspec["name"]] = {}
                                for arg in indspec["args"]:
                                    funcs[indspec["name"]][list(arg.keys())[0]] = list(
                                        arg.values()
                                    )[0]
        return funcs

    funcargs = getfuncargs()
    for funcname in tests.keys():
        with open(
            os.path.join(
                CURRENT_DIR, "..", "tests-kernels", "test_cpu" + funcname + ".py"
            ),
            "w",
        ) as f:
            f.write("import math\nimport ctypes\nfrom __init__ import lib, Error\n\n")
            num = 1
            for test in tests[funcname]:
                f.write("def test_cpu" + funcname + "_" + str(num) + "():\n")
                num += 1
                for arg, val in test["inargs"].items():
                    f.write(" " * 4 + arg + " = " + str(val) + "\n")
                    typename = funcargs[funcname][arg]
                    if "List" in typename:
                        count = typename.count("List")
                        typename = (
                            typename.replace("List", "")
                            .replace("[", "")
                            .replace("]", "")
                        )
                        if typename.endswith("_t"):
                            typename = typename[:-2]
                        if count == 1:
                            f.write(
                                " " * 4
                                + arg
                                + " = (ctypes.c_"
                                + typename
                                + "*len("
                                + arg
                                + "))(*"
                                + arg
                                + ")\n"
                            )
                        elif count == 2:
                            f.write(
                                " " * 4
                                + arg
                                + " = ctypes.pointer(ctypes.cast((ctypes.c_"
                                + typename
                                + "*len("
                                + arg
                                + "[0]))(*"
                                + arg
                                + "[0]),ctypes.POINTER(ctypes.c_"
                                + typename
                                + ")))\n"
                            )
                f.write(" " * 4 + "funcC = getattr(lib, '" + funcname + "')\n")
                f.write(" " * 4 + "funcC.restype = Error\n")
                f.write(
                    " " * 4
                    + "funcC.argtypes = "
                    + getctypelist(funcargs[funcname])
                    + "\n"
                )
                args = ""
                count = 0
                for arg in test["inargs"].keys():
                    if count == 0:
                        args += arg
                        count += 1
                    else:
                        args += ", " + arg
                if test["success"]:
                    f.write(" " * 4 + "ret_pass = funcC(" + args + ")\n")
                    for arg, val in test["outargs"].items():
                        f.write(" " * 4 + "out" + arg + " = " + str(val) + "\n")
                        if isinstance(val, list):
                            f.write(" " * 4 + "for i in range(len(out" + arg + ")):\n")
                            f.write(
                                " " * 8
                                + "assert math.isclose("
                                + arg
                                + "[i], out"
                                + arg
                                + "[i], rel_tol=0.0001)\n"
                            )
                        else:
                            f.write(" " * 4 + "assert " + arg + " == out" + arg + "\n")
                    f.write(" " * 4 + "assert not ret_pass.str\n")
                else:
                    f.write(" " * 4 + "assert funcC(" + args + ").str.contents\n")
                f.write("\n")


if __name__ == "__main__":
    genpykernels()
    tests = readspec()
    testpykernels(tests)
    testcpukernels(tests)
