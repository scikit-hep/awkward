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


def gettypename(spectype):
    typename = spectype.replace("List", "").replace("[", "").replace("]", "")
    if typename.endswith("_t"):
        typename = typename[:-2]
    return typename


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
                                        "{0} = {1}\n".format(
                                            childfunc["name"], indspec["name"]
                                        )
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
                                for x in indspec["args"]:
                                    for arg, val in x.items():
                                        if "Const[" in val:
                                            val = val.replace("Const[", "", 1).rstrip(
                                                "]"
                                            )
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


def getcudakernelslist():
    cudakernels = []
    for f in os.listdir(os.path.join(CURRENT_DIR, "..", "src", "cuda-kernels")):
        if os.path.isfile(os.path.join(CURRENT_DIR, "..", "src", "cuda-kernels", f)):
            if f.startswith("awkward_") and f.endswith(".cu"):
                cudakernels.append(f[:-3])
            elif f.startswith("manual_awkward_") and f.endswith(".cu"):
                cudakernels.append(f[len("manual_") : -3])
    return cudakernels


def genspectests(tests):
    genpykernels()
    print("Generating files for testing specification")
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
                                    + "assert {0}[:len(pytest_{0})] == pytest.approx(pytest_{0})\n".format(
                                        arg
                                    )
                                )
                            else:
                                f.write(
                                    " " * 4 + "assert {0} == pytest_{0}\n".format(arg)
                                )
                    else:
                        f.write(" " * 4 + "with pytest.raises(Exception):\n")
                        f.write(" " * 8 + "funcPy(" + args + ")\n")
                    f.write("\n")


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


def getctypelist(typedict):
    newctypes = []
    for typename in typedict.values():
        if "List" in typename:
            count = typename.count("List")
            typename = typename.replace("List", "").replace("[", "").replace("]", "")
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


def gencpukerneltests(tests):
    print("Generating files for testing CPU kernels")

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
                    if "List" in funcargs[funcname][arg]:
                        count = funcargs[funcname][arg].count("List")
                        typename = gettypename(funcargs[funcname][arg])
                        if count == 1:
                            f.write(
                                " " * 4
                                + "{0} = (ctypes.c_{1}*len({0}))(*{0})\n".format(
                                    arg, typename
                                )
                            )
                        elif count == 2:
                            f.write(
                                " " * 4
                                + "{0} = ctypes.pointer(ctypes.cast((ctypes.c_{1}*len({0}[0]))(*{0}[0]),ctypes.POINTER(ctypes.c_{1})))\n".format(
                                    arg, typename
                                )
                            )
                f.write(" " * 4 + "funcC = getattr(lib, '" + funcname + "')\n")
                f.write(" " * 4 + "funcC.restype = Error\n")
                f.write(
                    " " * 4
                    + "funcC.argtypes = {0}\n".format(getctypelist(funcargs[funcname]))
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
                                + "assert {0}[:len(pytest_{0})] == pytest.approx(pytest_{0})\n".format(
                                    arg
                                )
                            )
                        else:
                            f.write(" " * 4 + "assert {0} == pytest_{0}\n".format(arg))
                    f.write(" " * 4 + "assert not ret_pass.str\n")
                else:
                    f.write(" " * 4 + "assert funcC({0}).str.contents\n".format(args))
                f.write("\n")


def gencudakerneltests(tests):
    print("Generating files for testing CUDA kernels")

    funcargs = getfuncargs()
    cudakernels = getcudakernelslist()
    for funcname in tests.keys():
        if funcname in cudakernels:
            with open(
                os.path.join(
                    CURRENT_DIR,
                    "..",
                    "tests-cuda-kernels",
                    "test_cuda" + funcname + ".py",
                ),
                "w",
            ) as f:
                f.write(
                    "import ctypes\nimport cupy\nimport pytest\nfrom __init__ import lib, Error\n\n"
                )
                num = 1
                if tests[funcname] == []:
                    f.write(
                        "@pytest.mark.skip(reason='Unable to generate any tests for kernel')\n"
                    )
                    f.write("def test_cuda" + funcname + "_" + str(num) + "():\n")
                    f.write(
                        " " * 4
                        + "raise NotImplementedError('Unable to generate any tests for kernel')\n"
                    )
                for test in tests[funcname]:
                    f.write("def test_cuda" + funcname + "_" + str(num) + "():\n")
                    num += 1
                    for arg, val in test["inargs"].items():
                        if "List" in funcargs[funcname][arg]:
                            count = funcargs[funcname][arg].count(
                                "List"
                            )  # Might need later for ndim array
                            typename = gettypename(funcargs[funcname][arg])
                            f.write(
                                " " * 4
                                + "{0} = cupy.array({1}, dtype=cupy.{2})\n".format(
                                    arg, str(val), typename
                                )
                            )
                            f.write(
                                " " * 4
                                + "d_{0} = ctypes.cast({0}.data.ptr, ctypes.POINTER(ctypes.c_{1}))\n".format(
                                    arg, typename
                                )
                            )
                        else:
                            f.write(" " * 4 + "d_" + arg + " = " + str(val) + "\n")
                    f.write(" " * 4 + "funcC = getattr(lib, '" + funcname + "')\n")
                    f.write(" " * 4 + "funcC.restype = Error\n")
                    f.write(
                        " " * 4
                        + "funcC.argtypes = {0}\n".format(
                            getctypelist(funcargs[funcname])
                        )
                    )
                    args = ""
                    count = 0
                    for arg in funcargs[funcname].keys():
                        if count == 0:
                            args += "d_" + arg
                            count += 1
                        else:
                            args += ", d_" + arg
                    if test["success"]:
                        f.write(" " * 4 + "ret_pass = funcC(" + args + ")\n")
                        for arg, val in test["outargs"].items():
                            f.write(
                                " " * 4
                                + "pytest_{0} = cupy.array({1}, dtype=cupy.{2})\n".format(
                                    arg, str(val), gettypename(funcargs[funcname][arg])
                                )
                            )
                            if isinstance(val, list):
                                f.write(
                                    " " * 4
                                    + "for x in range(len(pytest_{0})):\n".format(arg)
                                )
                                f.write(
                                    " " * 8
                                    + "assert {0}[x] == pytest_{0}[x]\n".format(arg)
                                )
                            else:
                                f.write(
                                    " " * 4 + "assert {0} == pytest_{0}\n".format(arg)
                                )
                        f.write(" " * 4 + "assert not ret_pass.str\n")
                    else:
                        f.write(
                            " " * 4 + "assert funcC({0}).str.contents\n".format(args)
                        )
                    f.write("\n")


if __name__ == "__main__":
    tests = readspec()
    genspectests(tests)
    gencpukerneltests(tests)
    gencudakerneltests(tests)
