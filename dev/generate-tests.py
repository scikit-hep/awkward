# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import copy
import json
import os
from collections import OrderedDict
from itertools import product

import yaml
from parser_utils import PYGEN_BLACKLIST, SUCCESS_TEST_BLACKLIST, TEST_BLACKLIST

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


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


def typevalidates(testdict, arglist):
    for argdict in arglist:
        for arg, typename in argdict.items():
            if isinstance(testdict[arg], list):
                if testdict[arg] == []:
                    return False
                if not isinstance(testdict[arg][0], type(gettypeval(typename))):
                    return False
            else:
                if not isinstance(testdict[arg], type(gettypeval(typename))):
                    return False
    return True


def getdummyvalue(argdict, length):
    assert len(argdict) == 1
    return [gettypeval(list(argdict.values())[0])] * length


def dicttolist(outputdict, argdict):
    assert len(argdict) == 1
    typeval = gettypeval(list(argdict.values())[0])
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


def getargvals(funcname, arglist, rolelist, data, outparams):
    allvals = []
    with open(
        os.path.join(CURRENT_DIR, "..", "tests-spec", "kernels.py")
    ) as kernelfile:
        exec(kernelfile.read())
        instancedict = {}
        funcpassdict = OrderedDict()
        count = 0
        for roledict in rolelist:
            assert len(roledict) == 1
            for arg, role in roledict.items():
                funcpassdict[arg] = []
                if role == "default":
                    group = str(count)
                    assert group not in instancedict.keys()
                    instancedict[group] = [arg]
                    if arg in outparams:
                        funcpassdict[arg].append({})
                    else:
                        funcpassdict[arg].append(data["num"])
                    assert len(funcpassdict[arg]) == 1
                    count += 1
                else:
                    group = role[: role.find("-")]
                    if group not in instancedict.keys():
                        instancedict[group] = []
                    instancedict[group].append(arg)
                    if group not in data.keys() and group[:-1] in data.keys():
                        pseudogroup = copy.copy(group[:-1])
                    elif group in data.keys():
                        pseudogroup = copy.copy(group)
                    role = pseudogroup + role[role.find("-") :]
                    for x in range(len(data[pseudogroup])):
                        funcpassdict[arg].append(data[pseudogroup][x][role])

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
            funcPy = eval(funcname)

            intests = OrderedDict()
            outtests = OrderedDict()
            tempdict = {}
            try:
                funcPy(**temp)
                if (funcname not in SUCCESS_TEST_BLACKLIST) or any(
                    funcname in x for x in SUCCESS_TEST_BLACKLIST
                ):
                    for arg in temp.keys():
                        if arg in outparams:
                            assert isinstance(temp[arg], dict)
                            temparglist = dicttolist(
                                temp[arg],
                                next(
                                    argmap
                                    for argmap in arglist
                                    if list(argmap.keys())[0] == arg
                                ),
                            )
                            intests[arg] = getdummyvalue(
                                next(
                                    argmap
                                    for argmap in arglist
                                    if list(argmap.keys())[0] == arg
                                ),
                                len(temparglist),
                            )
                            outtests[arg] = temparglist
                        else:
                            intests[arg] = temp[arg]
                    tempdict["outargs"] = copy.deepcopy(outtests)
                    tempdict["success"] = True
            except ValueError:
                for arg in temp.keys():
                    if arg in outparams:
                        intests[arg] = getdummyvalue(
                            next(
                                argmap
                                for argmap in arglist
                                if list(argmap.keys())[0] == arg
                            ),
                            len(temp[arg]),
                        )
                    else:
                        intests[arg] = temp[arg]
                tempdict["success"] = False
            tempdict["inargs"] = copy.deepcopy(intests)
            if typevalidates(tempdict["inargs"], arglist):
                allvals.append(tempdict)

    return allvals


def gettests():
    tests = {}
    with open(
        os.path.join(CURRENT_DIR, "..", "kernel-specification", "samples.json")
    ) as testjson:
        data = json.load(testjson)
        genpykernels()
        print("Generating test cases")
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
                        if "specializations" in indspec.keys():
                            for childfunc in indspec["specializations"]:
                                if (
                                    "roles" in indspec.keys()
                                    and (
                                        "pointer"
                                        not in [
                                            list(x.values())[0]
                                            for x in indspec["roles"]
                                        ]
                                    )
                                    and indspec["name"] not in TEST_BLACKLIST
                                    and indspec["name"] not in PYGEN_BLACKLIST
                                ):
                                    tests[childfunc["name"]] = getargvals(
                                        childfunc["name"],
                                        childfunc["args"],
                                        indspec["roles"],
                                        data,
                                        indspec["outparams"],
                                    )
                                else:
                                    tests[childfunc["name"]] = []
                        else:
                            if (
                                "roles" in indspec.keys()
                                and "pointer"
                                not in [list(x.values())[0] for x in indspec["roles"]]
                                and indspec["name"] not in TEST_BLACKLIST
                                and indspec["name"] not in PYGEN_BLACKLIST
                            ):
                                tests[indspec["name"]] = getargvals(
                                    indspec["name"],
                                    indspec["args"],
                                    indspec["roles"],
                                    data,
                                    indspec["outparams"],
                                )
                            else:
                                tests[indspec["name"]] = []
    return tests


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


def getfuncargs(tests):
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
                    if "def " in indspec["definition"]:
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

    funcargs = getfuncargs(tests)
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

    funcargs = getfuncargs(tests)
    cudakernels = getcudakernelslist()
    funcnames = getfuncnames()
    cudafuncnames = {funcname: funcnames[funcname] for funcname in cudakernels}
    for funcname in tests.keys():
        if (funcname in cudakernels) or any(
            funcname in x for x in cudafuncnames.values()
        ):
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
                        f.write(" " * 4 + "assert funcC({0}).str\n".format(args))
                    f.write("\n")


if __name__ == "__main__":
    tests = gettests()
    genspectests(tests)
    gencpukerneltests(tests)
    gencudakerneltests(tests)
