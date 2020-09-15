# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import copy
import json
import os
from collections import OrderedDict
from itertools import product

import yaml
from numpy import uint8
from parser_utils import (PYGEN_BLACKLIST, SUCCESS_TEST_BLACKLIST,
                          TEST_BLACKLIST)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


class Argument(object):
    def __init__(self, name, typename, direction, role="default"):
        self.name = name
        self.typename = typename
        self.direction = direction
        self.role = role


class Specification(object):
    def __init__(self, spec, testdata, blacklisted):
        self.name = spec["name"]
        self.args = []
        for arg in spec["args"]:
            self.args.append(
                Argument(
                    arg["name"],
                    arg["type"],
                    arg["direction"],
                    arg["role"] if "role" in arg.keys() else "default",
                )
            )
        if blacklisted:
            self.tests = []
        else:
            self.tests = self.gettests(testdata)

    def validateoverflow(self, testvals):
        flag = True
        for arg in self.args:
            if "uint" in arg.typename and (
                any(n < 0 for n in testvals["inargs"][arg.name])
                or (
                    "outargs" in testvals.keys()
                    and arg.name in testvals["outargs"].keys()
                    and any(n < 0 for n in testvals["outargs"][arg.name])
                )
            ):
                flag = False
        return flag

    def dicttolist(self, outputdict, typename):
        typeval = gettypeval(typename)
        vallist = []
        count = 0
        for num in sorted(outputdict):
            if num == count:
                vallist.append(outputdict[num])
            else:
                while num != count:
                    count += 1
                    vallist.append(typeval)
                vallist.append(outputdict[num])
            count += 1
        return vallist

    def getdummyvalue(self, typename, length):
        return [gettypeval(typename)] * length

    def typevalidates(self, testdict, arglist):
        for arg in arglist:
            if isinstance(testdict[arg.name], list):
                if testdict[arg.name] == []:
                    return False
                if not isinstance(
                    testdict[arg.name][0], type(gettypeval(arg.typename))
                ):
                    return False
            else:
                if not isinstance(testdict[arg.name], type(gettypeval(arg.typename))):
                    return False
        return True

    def gettests(self, testdata):
        allvals = []
        with open(
            os.path.join(CURRENT_DIR, "..", "tests-spec", "kernels.py")
        ) as kernelfile:
            wrap_exec(kernelfile.read(), globals(), locals())
            instancedict = {}
            funcpassdict = OrderedDict()
            count = 0
            for arg in self.args:
                funcpassdict[arg.name] = []
                if arg.role == "default":
                    group = str(count)
                    assert group not in instancedict.keys()
                    instancedict[group] = [arg.name]
                    if arg.direction == "out":
                        funcpassdict[arg.name].append({})
                    else:
                        funcpassdict[arg.name].append(testdata["num"])
                    assert len(funcpassdict[arg.name]) == 1
                    count += 1
                else:
                    group = arg.role[: arg.role.find("-")]
                    if group not in instancedict.keys():
                        instancedict[group] = []
                    instancedict[group].append(arg.name)
                    if group not in testdata.keys() and group[:-1] in testdata.keys():
                        pseudogroup = copy.copy(group[:-1])
                    elif group in testdata.keys():
                        pseudogroup = copy.copy(group)
                    role = pseudogroup + arg.role[arg.role.find("-") :]
                    for x in range(len(testdata[pseudogroup])):
                        funcpassdict[arg.name].append(testdata[pseudogroup][x][role])

            instancedictlist = list(instancedict.keys())

            combinations = []
            for name in instancedictlist:
                temp = []
                for arg in instancedict[name]:
                    temp.append(funcpassdict[arg])
                combinations.append(zip(*temp))

            for x in product(*combinations):
                origtemp = OrderedDict()
                for groupName, t in zip(instancedictlist, x):
                    for key, value in zip(instancedict[groupName], t):
                        origtemp[key] = value

                temp = copy.deepcopy(origtemp)
                funcPy = wrap_eval(self.name, globals(), locals())

                intests = OrderedDict()
                outtests = OrderedDict()
                tempdict = {}
                try:
                    funcPy(**temp)
                    if self.name not in SUCCESS_TEST_BLACKLIST or any(
                        self.name in x for x in SUCCESS_TEST_BLACKLIST
                    ):
                        for arg in self.args:
                            if arg.direction == "out":
                                assert isinstance(temp[arg.name], dict)
                                temparglist = self.dicttolist(
                                    temp[arg.name], arg.typename
                                )
                                intests[arg.name] = self.getdummyvalue(
                                    arg.typename, len(temparglist)
                                )
                                outtests[arg.name] = temparglist
                            else:
                                intests[arg.name] = temp[arg.name]
                        tempdict["outargs"] = copy.deepcopy(outtests)
                        tempdict["success"] = True
                except ValueError:
                    for arg in self.args:
                        if arg.direction == "out":
                            intests[arg.name] = self.getdummyvalue(
                                arg.typename, len(temp[arg.name])
                            )
                        else:
                            intests[arg.name] = temp[arg.name]
                    tempdict["success"] = False
                tempdict["inargs"] = copy.deepcopy(intests)
                if self.typevalidates(
                    tempdict["inargs"], self.args
                ) and self.validateoverflow(tempdict):
                    allvals.append(tempdict)

        return allvals


def readspec():
    genpykernels()
    specdict = {}
    with open(
        os.path.join(CURRENT_DIR, "..", "kernel-specification", "samples.json")
    ) as testjson:
        data = json.load(testjson)
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
                        if "def " in indspec["definition"]:
                            if "specializations" in indspec.keys():
                                for childfunc in indspec["specializations"]:
                                    specdict[childfunc["name"]] = Specification(
                                        childfunc,
                                        data,
                                        (indspec["name"] in TEST_BLACKLIST)
                                        or (indspec["name"] in PYGEN_BLACKLIST),
                                    )
                            else:
                                specdict[indspec["name"]] = Specification(
                                    indspec,
                                    data,
                                    (indspec["name"] in TEST_BLACKLIST)
                                    or (indspec["name"] in PYGEN_BLACKLIST),
                                )
    return specdict


def wrap_exec(string, globs, locs):
    exec(string, globs, locs)


def wrap_eval(string, globs, locs):
    return eval(string, globs, locs)


def gettypename(spectype):
    typename = spectype.replace("List", "").replace("[", "").replace("]", "")
    if typename.endswith("_t"):
        typename = typename[:-2]
    return typename


def getfuncnames():
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
                    funcs[indspec["name"]] = []
                    if "specializations" in indspec.keys():
                        for childfunc in indspec["specializations"]:
                            funcs[indspec["name"]].append(childfunc["name"])
    return funcs


def genpykernels():
    print("Generating Python kernels")
    prefix = """
from numpy import uint8
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


def gettypeval(typename):
    if "int" in typename:
        typeval = 123
    elif "bool" in typename:
        typeval = True
    elif "double" in typename or "float" in typename:
        typeval = 123.0
    else:
        raise ValueError("Unknown type encountered")
    return typeval


def getcudakernelslist():
    cudakernels = []
    for f in os.listdir(os.path.join(CURRENT_DIR, "..", "src", "cuda-kernels")):
        if os.path.isfile(os.path.join(CURRENT_DIR, "..", "src", "cuda-kernels", f)):
            if f.startswith("awkward_") and f.endswith(".cu"):
                cudakernels.append(f[:-3])
            elif f.startswith("manual_awkward_") and f.endswith(".cu"):
                cudakernels.append(f[len("manual_") : -3])
    return cudakernels


def genspectests(specdict):
    print("Generating files for testing specification")
    for spec in specdict.values():
        with open(
            os.path.join(
                CURRENT_DIR, "..", "tests-spec", "test_py" + spec.name + ".py"
            ),
            "w",
        ) as f:
            f.write("import pytest\nimport kernels\n\n")
            num = 1
            if spec.tests == []:
                f.write(
                    "@pytest.mark.skip(reason='Unable to generate any tests for kernel')\n"
                )
                f.write("def test_py" + spec.name + "_" + str(num) + "():\n")
                f.write(
                    " " * 4
                    + "raise NotImplementedError('Unable to generate any tests for kernel')\n"
                )
            else:
                for test in spec.tests:
                    f.write("def test_py" + spec.name + "_" + str(num) + "():\n")
                    num += 1
                    args = ""
                    for arg, val in test["inargs"].items():
                        f.write(" " * 4 + arg + " = " + str(val) + "\n")
                    f.write(
                        " " * 4 + "funcPy = getattr(kernels, '" + spec.name + "')\n"
                    )
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


def remove_const(typename):
    if "Const[" in typename:
        typename = typename.replace("Const[", "", 1).rstrip("]")
    return typename


def getctypelist(arglist):
    newctypes = []
    for arg in arglist:
        typename = remove_const(arg.typename)
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


def gencpukerneltests(specdict):
    print("Generating files for testing CPU kernels")

    for spec in specdict.values():
        with open(
            os.path.join(
                CURRENT_DIR, "..", "tests-cpu-kernels", "test_cpu" + spec.name + ".py"
            ),
            "w",
        ) as f:
            f.write("import ctypes\nimport pytest\nfrom __init__ import lib, Error\n\n")
            num = 1
            if spec.tests == []:
                f.write(
                    "@pytest.mark.skip(reason='Unable to generate any tests for kernel')\n"
                )
                f.write("def test_cpu" + spec.name + "_" + str(num) + "():\n")
                f.write(
                    " " * 4
                    + "raise NotImplementedError('Unable to generate any tests for kernel')\n"
                )
            for test in spec.tests:
                f.write("def test_cpu" + spec.name + "_" + str(num) + "():\n")
                num += 1
                for arg, val in test["inargs"].items():
                    f.write(" " * 4 + arg + " = " + str(val) + "\n")
                    typename = remove_const(
                        next(
                            argument for argument in spec.args if argument.name == arg
                        ).typename
                    )
                    if "List" in typename:
                        count = typename.count("List")
                        typename = gettypename(typename)
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
                f.write(" " * 4 + "funcC = getattr(lib, '" + spec.name + "')\n")
                f.write(" " * 4 + "funcC.restype = Error\n")
                f.write(
                    " " * 4 + "funcC.argtypes = {0}\n".format(getctypelist(spec.args))
                )
                args = ""
                count = 0
                for arg in spec.args:
                    if count == 0:
                        args += arg.name
                        count += 1
                    else:
                        args += ", " + arg.name
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


def gencudakerneltests(specdict):
    print("Generating files for testing CUDA kernels")

    cudakernels = getcudakernelslist()
    funcnames = getfuncnames()
    cudafuncnames = {funcname: funcnames[funcname] for funcname in cudakernels}
    for spec in specdict.values():
        if (spec.name in cudakernels) or any(
            spec.name in x for x in cudafuncnames.values()
        ):
            with open(
                os.path.join(
                    CURRENT_DIR,
                    "..",
                    "tests-cuda-kernels",
                    "test_cuda" + spec.name + ".py",
                ),
                "w",
            ) as f:
                f.write(
                    "import ctypes\nimport cupy\nimport pytest\nfrom __init__ import lib, Error\n\n"
                )
                num = 1
                if spec.tests == []:
                    f.write(
                        "@pytest.mark.skip(reason='Unable to generate any tests for kernel')\n"
                    )
                    f.write("def test_cuda" + spec.name + "_" + str(num) + "():\n")
                    f.write(
                        " " * 4
                        + "raise NotImplementedError('Unable to generate any tests for kernel')\n"
                    )
                for test in spec.tests:
                    f.write("def test_cuda" + spec.name + "_" + str(num) + "():\n")
                    num += 1
                    for arg, val in test["inargs"].items():
                        typename = remove_const(
                            next(
                                argument
                                for argument in spec.args
                                if argument.name == arg
                            ).typename
                        )
                        if "List" in typename:
                            count = typename.count(
                                "List"
                            )  # Might need later for ndim array
                            typename = gettypename(typename)
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
                    f.write(" " * 4 + "funcC = getattr(lib, '" + spec.name + "')\n")
                    f.write(" " * 4 + "funcC.restype = Error\n")
                    f.write(
                        " " * 4
                        + "funcC.argtypes = {0}\n".format(getctypelist(spec.args))
                    )
                    args = ""
                    count = 0
                    for arg in spec.args:
                        if count == 0:
                            args += "d_" + arg.name
                            count += 1
                        else:
                            args += ", d_" + arg.name
                    if test["success"]:
                        f.write(" " * 4 + "ret_pass = funcC(" + args + ")\n")
                        for arg, val in test["outargs"].items():
                            f.write(
                                " " * 4
                                + "pytest_{0} = cupy.array({1}, dtype=cupy.{2})\n".format(
                                    arg,
                                    str(val),
                                    gettypename(
                                        next(
                                            argument.typename
                                            for argument in spec.args
                                            if argument.name == arg
                                        )
                                    ),
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
                        f.write(" " * 4 + "assert funcC({0}).str\n".format(args))
                    f.write("\n")


if __name__ == "__main__":
    specdict = readspec()
    genspectests(specdict)
    gencpukerneltests(specdict)
    gencudakerneltests(specdict)
