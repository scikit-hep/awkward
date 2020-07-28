# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import os
import subprocess
from collections import OrderedDict

import yaml

from parser_utils import pytype

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
CPU_KERNEL_SO = ""
for root, _, files in os.walk(CURRENT_DIR[:-3]):
    for filename in files:
        if filename.endswith("libawkward-cpu-kernels.so"):
            CPU_KERNEL_SO = os.path.join(root, filename)
            break


def genpykernels():
    prefix = """

kMaxInt64  = 9223372036854775806
kSliceNone = kMaxInt64 + 1

"""
    with open(os.path.join(CURRENT_DIR, "spec.yaml")) as infile:
        spec = yaml.safe_load(infile)["kernels"]
        with open(os.path.join(CURRENT_DIR, "..", "tests", "kernels.py"), "w") as outfile:
            outfile.write(prefix)
            for func in spec:
                if "def " in func["definition"]:
                    outfile.write(func["definition"] + "\n")
                    if "specializations" in func.keys():
                        for childfunc in func["specializations"]:
                            outfile.write(
                                childfunc["name"] + " = " + func["name"] + "\n"
                            )
                        outfile.write("\n\n")


def readspec():
    funcs = {}
    with open(os.path.join(CURRENT_DIR, "spec.yaml")) as infile:
        spec = yaml.safe_load(infile)["kernels"]
        for func in spec:
            if (
                "def " in func["definition"]
                and "tests" in func.keys()
                and func["tests"] is not None
            ):
                if "specializations" in func.keys():
                    for childfunc in func["specializations"]:
                        funcs[childfunc["name"]] = []
                        for test in func["tests"]:
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
                                    testinfo["outargs"].update(test["results"])
                                funcs[childfunc["name"]].append(testinfo)
                else:
                    funcs[func["name"]] = []
                    for test in func["tests"]:
                        # Check if test has correct types
                        flag = True
                        count = 0
                        for arg, val in test["args"].items():
                            spectype = pytype(
                                func["args"][count][arg]
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
                            funcs[func["name"]].append(testinfo)
    return funcs


def testpykernels(tests):
    with open(os.path.join(CURRENT_DIR, "..", "tests", "test_pykernels.py"), "w") as f:
        f.write("import tests.kernels\n\n")
        for funcname in tests.keys():
            num = 1
            for test in tests[funcname]:
                f.write("def test_" + funcname + "_" + str(num) + "():\n")
                num += 1
                args = ""
                for arg, val in test["inargs"].items():
                    f.write(" " * 4 + arg + " = " + str(val) + "\n")
                f.write(" " * 4 + "funcPy = getattr(tests.kernels, '" + funcname + "')\n")
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
                    f.write(" " * 4 + "try:\n")
                    f.write(" " * 8 + "funcPy(" + args + ")\n")
                    f.write(" " * 8 + "assert False\n")
                    f.write(" " * 4 + "except:\n")
                    f.write(" " * 8 + "pass\n")
                f.write("\n")


def testcpukernels(tests):
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
        with open(os.path.join(CURRENT_DIR, "spec.yaml")) as infile:
            spec = yaml.safe_load(infile)["kernels"]
            for func in spec:
                if (
                    "def " in func["definition"]
                    and "tests" in func.keys()
                    and func["tests"] is not None
                ):
                    if "specializations" in func.keys():
                        for childfunc in func["specializations"]:
                            funcs[childfunc["name"]] = {}
                            for arg in childfunc["args"]:
                                funcs[childfunc["name"]][list(arg.keys())[0]] = list(
                                    arg.values()
                                )[0]
                    else:
                        funcs[func["name"]] = {}
                        for arg in func["args"]:
                            funcs[func["name"]][list(arg.keys())[0]] = list(
                                arg.values()
                            )[0]
        return funcs

    if CPU_KERNEL_SO == "":
        raise AssertionError("Unable to find libawkward-cpu-kernels.so")
    with open(os.path.join(CURRENT_DIR, "..", "tests", "test_cpukernels.py"), "w") as f:
        f.write("import math\nimport ctypes\n\n")
        f.write(
            """
class Error(ctypes.Structure):
    _fields_ = [
        ("str", ctypes.POINTER(ctypes.c_char)),
        ("identity", ctypes.c_int64),
        ("attempt", ctypes.c_int64),
        ("pass_through", ctypes.c_bool),
    ]
"""
        )
        f.write("lib = ctypes.CDLL('" + CPU_KERNEL_SO + "')\n\n")
        funcargs = getfuncargs()
        for funcname in tests.keys():
            num = 1
            for test in tests[funcname]:
                f.write("def test_" + funcname + "_" + str(num) + "():\n")
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
