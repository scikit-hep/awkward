# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import os
import re
from collections import OrderedDict

import yaml

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def pytype(cpptype):
    cpptype = cpptype.replace("*", "")
    if re.match("u?int\d{1,2}(_t)?", cpptype) is not None:
        return "int"
    elif cpptype == "double":
        return "float"
    else:
        return cpptype


def genpykernels():
    print("Generating Python kernels")
    prefix = """

kMaxInt64  = 9223372036854775806
kSliceNone = kMaxInt64 + 1

"""
    with open(
        os.path.join(CURRENT_DIR, "..", "kernel-specification", "kernelnames.yml")
    ) as infile:
        mainspec = yaml.safe_load(infile)["kernels"]
        with open(
            os.path.join(CURRENT_DIR, "..", "tests-spec", "kernels.py"), "w"
        ) as outfile:
            outfile.write(prefix)
            for filedir in mainspec.values():
                for relpath in filedir.values():
                    with open(
                        os.path.join(CURRENT_DIR, "..", "kernel-specification", relpath)
                    ) as specfile:
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
    with open(
        os.path.join(CURRENT_DIR, "..", "kernel-specification", "kernelnames.yml")
    ) as infile:
        mainspec = yaml.safe_load(infile)["kernels"]
        for filedir in mainspec.values():
            for relpath in filedir.values():
                with open(
                    os.path.join(CURRENT_DIR, "..", "kernel-specification", relpath)
                ) as specfile:
                    indspec = yaml.safe_load(specfile)[0]
                    if indspec.get("tests") is None:
                        indspec["tests"] = []
                    if "tests" in indspec.keys():
                        if "specializations" in indspec.keys():
                            for childfunc in indspec["specializations"]:
                                funcs[childfunc["name"]] = []
                                for test in indspec["tests"]:
                                    # Check if test has correct types
                                    flag = True
                                    for x in childfunc["args"]:
                                        for arg, val in x.items():
                                            if "Const[" in val:
                                                val = val.replace(
                                                    "Const[", "", 1
                                                ).rstrip("]")
                                            spectype = pytype(
                                                val.replace("List", "")
                                                .replace("[", "")
                                                .replace("]", "")
                                            )
                                            testval = test["args"][arg]
                                            while isinstance(testval, list):
                                                if len(testval) == 0:
                                                    testval = None
                                                else:
                                                    testval = testval[0]
                                            if type(testval) != eval(spectype):
                                                flag = False
                                            elif test["successful"] and (
                                                "U32" in childfunc["name"]
                                                or "U64" in childfunc["name"]
                                                or (
                                                    arg in test["results"].keys()
                                                    and -1 in test["results"][arg]
                                                )
                                            ):
                                                flag = False
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
                            funcs[indspec["name"]] = []
                            for test in indspec["tests"]:
                                # Check if test has correct types
                                flag = True
                                for test in indspec["tests"]:
                                    # Check if test has correct types
                                    flag = True
                                    for x in indspec["args"]:
                                        for arg, val in x.items():
                                            if "Const[" in val:
                                                val = val.replace(
                                                    "Const[", "", 1
                                                ).rstrip("]")
                                            spectype = pytype(
                                                val.replace("List", "")
                                                .replace("[", "")
                                                .replace("]", "")
                                            )
                                            testval = test["args"][arg]
                                            while isinstance(testval, list):
                                                testval = testval[0]
                                            if type(testval) != eval(spectype):
                                                flag = False
                                            elif test["successful"] and (
                                                "U32" in indspec["name"]
                                                or "U64" in indspec["name"]
                                                or (
                                                    arg in test["results"].keys()
                                                    and -1 in test["results"][arg]
                                                )
                                            ):
                                                flag = False
                                if flag:
                                    testinfo = {}
                                    testinfo["inargs"] = OrderedDict()
                                    testinfo["inargs"].update(test["args"])
                                    testinfo["success"] = test["successful"]
                                    if testinfo["success"]:
                                        testinfo["outargs"] = OrderedDict()
                                        testinfo["outargs"].update(test["results"])
                                    funcs[indspec["name"]].append(testinfo)
                    else:
                        funcs[indspec["name"]] = []
    return funcs


def testpykernels(tests):
    genpykernels()
    print("Generating file for testing python kernels")
    for funcname in tests.keys():
        with open(
            os.path.join(CURRENT_DIR, "..", "tests-spec", "test_py" + funcname + ".py"),
            "w",
        ) as f:
            f.write("import pytest\nimport kernels\n\n")
            num = 1
            if tests[funcname] == []:
                f.write(
                    "@pytest.mark.skip(reason='Unable to generate any tests for kernel')\n"
                )
                f.write("def test_py" + funcname + "_" + str(num) + "():\n")
                f.write(
                    " " * 4
                    + "raise NotImplementedError('Unable to generate any tests for kernel')\n"
                )
            else:
                for test in tests[funcname]:
                    f.write("def test_py" + funcname + "_" + str(num) + "():\n")
                    num += 1
                    args = ""
                    for arg, val in test["inargs"].items():
                        f.write(" " * 4 + arg + " = " + str(val) + "\n")
                    f.write(" " * 4 + "funcPy = getattr(kernels, '" + funcname + "')\n")
                    count = 0
                    for arg in test["inargs"].keys():
                        if count == 0:
                            args += arg + "=" + arg
                            count += 1
                        else:
                            args += ", " + arg + "=" + arg
                    if test["success"]:
                        f.write(" " * 4 + "funcPy" + "(" + args + ")\n")
                        for arg, val in test["outargs"].items():
                            f.write(" " * 4 + "pytest_" + arg + " = " + str(val) + "\n")
                            if isinstance(val, list):
                                f.write(
                                    " " * 4
                                    + "assert "
                                    + arg
                                    + "[:len(pytest_"
                                    + arg
                                    + ")] == pytest.approx(pytest_"
                                    + arg
                                    + ")\n"
                                )
                            else:
                                f.write(
                                    " " * 4
                                    + "assert "
                                    + arg
                                    + " == pytest_"
                                    + arg
                                    + "\n"
                                )
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
        with open(
            os.path.join(CURRENT_DIR, "..", "kernel-specification", "kernelnames.yml")
        ) as infile:
            mainspec = yaml.safe_load(infile)["kernels"]
            for filedir in mainspec.values():
                for relpath in filedir.values():
                    with open(
                        os.path.join(CURRENT_DIR, "..", "kernel-specification", relpath)
                    ) as specfile:
                        indspec = yaml.safe_load(specfile)[0]
                        if (
                            "def " in indspec["definition"]
                            and "tests" in indspec.keys()
                            and indspec["tests"] is not None
                        ):
                            if "specializations" in indspec.keys():
                                for childfunc in indspec["specializations"]:
                                    funcs[childfunc["name"]] = OrderedDict()
                                    if tests[childfunc["name"]] != []:
                                        for arg in childfunc["args"]:
                                            typename = list(arg.values())[0]
                                            if "Const[" in typename:
                                                typename = typename.replace(
                                                    "Const[", "", 1
                                                ).rstrip("]")
                                            funcs[childfunc["name"]][
                                                list(arg.keys())[0]
                                            ] = typename

                            else:
                                funcs[indspec["name"]] = OrderedDict()
                                if tests[indspec["name"]] != []:
                                    for arg in indspec["args"]:
                                        typename = list(arg.values())[0]
                                        if "Const[" in typename:
                                            typename = typename.replace(
                                                "Const[", "", 1
                                            ).rstrip("]")
                                        funcs[indspec["name"]][
                                            list(arg.keys())[0]
                                        ] = typename

        return funcs

    funcargs = getfuncargs()
    for funcname in tests.keys():
        with open(
            os.path.join(
                CURRENT_DIR, "..", "tests-cpu-kernels", "test_cpu" + funcname + ".py"
            ),
            "w",
        ) as f:
            f.write("import ctypes\nimport pytest\nfrom __init__ import lib, Error\n\n")
            num = 1
            if tests[funcname] == []:
                f.write(
                    "@pytest.mark.skip(reason='Unable to generate any tests for kernel')\n"
                )
                f.write("def test_cpu" + funcname + "_" + str(num) + "():\n")
                f.write(
                    " " * 4
                    + "raise NotImplementedError('Unable to generate any tests for kernel')\n"
                )
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
                for arg in funcargs[funcname].keys():
                    if count == 0:
                        args += arg
                        count += 1
                    else:
                        args += ", " + arg
                if test["success"]:
                    f.write(" " * 4 + "ret_pass = funcC(" + args + ")\n")
                    for arg, val in test["outargs"].items():
                        f.write(" " * 4 + "pytest_" + arg + " = " + str(val) + "\n")
                        if isinstance(val, list):
                            f.write(
                                " " * 4
                                + "assert "
                                + arg
                                + "[:len(pytest_"
                                + arg
                                + ")] == pytest.approx(pytest_"
                                + arg
                                + ")\n"
                            )
                        else:
                            f.write(
                                " " * 4 + "assert " + arg + " == pytest_" + arg + "\n"
                            )
                    f.write(" " * 4 + "assert not ret_pass.str\n")
                else:
                    f.write(" " * 4 + "assert funcC(" + args + ").str.contents\n")
                f.write("\n")


if __name__ == "__main__":
    tests = readspec()
    testpykernels(tests)
    testcpukernels(tests)
