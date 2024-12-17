# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import copy
import datetime
import json
import os
import shutil
import time
from collections import OrderedDict
from itertools import product

import numpy as np
import yaml
from numpy import uint8  # noqa: F401 (used in evaluated strings)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def reproducible_datetime():
    import sys

    timestamp = int(os.environ.get("SOURCE_DATE_EPOCH", time.time()))

    if sys.version_info >= (3, 11):
        build_date = datetime.datetime.fromtimestamp(timestamp, tz=datetime.UTC)
    else:
        build_date = datetime.datetime.utcfromtimestamp(
            int(os.environ.get("SOURCE_DATE_EPOCH", time.time()))
        )
    return build_date.isoformat().replace("T", " AT ")[:22]


class Argument:
    __slots__ = ("direction", "name", "role", "typename")

    def __init__(self, name, typename, direction, role="default"):
        self.name = name
        self.typename = typename
        self.direction = direction
        self.role = role


no_role_kernels = [
    "awkward_NumpyArray_sort_asstrings_uint8",
    "awkward_argsort",
    "awkward_sort",
]


class Specification:
    def __init__(self, templatized_kernel_name, spec, testdata, blacklisted):
        self.templatized_kernel_name = templatized_kernel_name
        self.name = spec["name"]
        self.args = []
        for arg in spec["args"]:
            self.args.append(
                Argument(
                    arg["name"],
                    arg["type"],
                    arg["dir"],
                    arg["role"] if "role" in arg.keys() else "default",
                )
            )
        if blacklisted:
            self.tests = []
        elif templatized_kernel_name in no_role_kernels:
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
            funcPy = eval(self.name, globals(), locals())

            intests = OrderedDict()
            outtests = OrderedDict()
            tempdict = {}
            try:
                funcPy(**temp)
                for arg in self.args:
                    if arg.direction == "out":
                        assert isinstance(temp[arg.name], dict)
                        temparglist = self.dicttolist(temp[arg.name], arg.typename)
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
    specdict = {}
    specdict_unit = {}
    with open(os.path.join(CURRENT_DIR, "..", "kernel-specification.yml")) as f:
        loadfile = yaml.load(f, Loader=yaml.CSafeLoader)

    indspec = loadfile["kernels"]
    with open(os.path.join(CURRENT_DIR, "..", "kernel-test-data.json")) as f:
        data = json.load(f)["tests"]

    for spec in indspec:
        for childfunc in spec["specializations"]:
            specdict_unit[childfunc["name"]] = Specification(
                spec["name"],
                childfunc,
                data,
                not spec["automatic-tests"],
            )
        if "def " in spec["definition"]:
            for childfunc in spec["specializations"]:
                specdict[childfunc["name"]] = Specification(
                    spec["name"],
                    childfunc,
                    data,
                    not spec["automatic-tests"],
                )
    return specdict, specdict_unit


def getdtypes(args):
    dtypes = []
    for arg in args:
        typename = remove_const(arg.typename)
        if "List" in typename:
            count = typename.count("List")
            typename = gettypename(typename)
            if typename == "bool":
                typename = typename + "_"
            if typename == "float":
                typename = typename + "32"
            if count == 1:
                dtypes.append("cupy." + typename)
            elif count == 2:
                dtypes.append("cupy." + typename)
    return dtypes


def gettype(arg, args):
    typename = remove_const(
        next(argument for argument in args if argument.name == arg).typename
    )
    return typename


def checkuint(test_args, args):
    flag = True
    for arg, val in test_args:
        typename = gettype(arg, args)
        if "List[uint" in typename and (any(n < 0 for n in val)):
            flag = False
    return flag


def checkintrange(test_args, error, args):
    flag = True
    if not error:
        for arg, val in test_args:
            typename = gettype(arg, args)
            if "int" in typename or "uint" in typename:
                dtype = gettypename(typename)
                min_val, max_val = np.iinfo(dtype).min, np.iinfo(dtype).max
                if "List[List" in typename:
                    for row in val:
                        for data in row:
                            if not (min_val <= data <= max_val):
                                flag = False
                elif "List" in typename:
                    for data in val:
                        if not (min_val <= data <= max_val):
                            flag = False
                else:
                    if not (min_val <= val <= max_val):
                        flag = False
    return flag


def unittestmap():
    with open(os.path.join(CURRENT_DIR, "..", "kernel-test-data.json")) as f:
        data = json.load(f)["unit-tests"]
    unit_tests_map = {}
    for function in data:
        tests = function["tests"]
        status = function["status"]
        unit_tests_map[function["name"]] = {"tests": tests, "status": status}
    return unit_tests_map


def getunittests(test_inputs, test_outputs):
    unit_tests = {**test_outputs, **test_inputs}
    return unit_tests


def gettypename(spectype):
    typename = spectype.replace("List", "").replace("[", "").replace("]", "")
    if typename.endswith("_t"):
        typename = typename[:-2]
    return typename


def genpykernels():
    print("Generating Python kernels")
    prefix = """
import numpy
from numpy import uint8
kMaxInt64  = 9223372036854775806
kSliceNone = kMaxInt64 + 1

def awkward_regularize_rangeslice(
    start, stop, posstep, hasstart, hasstop, length,
):
    if posstep:
        if not hasstart:         start = 0
        elif start < 0:          start += length
        if start < 0:            start = 0
        if start > length:       start = length

        if not hasstop:          stop = length
        elif stop < 0:           stop += length
        if stop < 0:             stop = 0
        if stop > length:        stop = length
        if stop < start:         stop = start

    else:
        if not hasstart:         start = length - 1
        elif start < 0:          start += length
        if start < -1:           start = -1
        if start > length - 1:   start = length - 1

        if not hasstop:          stop = -1
        elif stop < 0:           stop += length
        if stop < -1:            stop = -1
        if stop > length - 1:    stop = length - 1
        if stop > start:         stop = start
    return start, stop

def awkward_ListArray_combinations_step(
    tocarry, toindex, fromindex, j, stop, n, replacement
):
    while fromindex[j] < stop:
        if replacement:
            for k in range(j + 1, n):
                fromindex[k] = fromindex[j]
        else:
            for k in range(j + 1, n):
                fromindex[k] = fromindex[j] + (k - j)

        if j + 1 == n:
            for k in range(n):
                tocarry[k][toindex[k]] = fromindex[k]
                toindex[k] += 1
        else:
            awkward_ListArray_combinations_step(tocarry, toindex, fromindex, j + 1, stop, n, replacement)

        fromindex[j] += 1

"""

    tests_spec = os.path.join(CURRENT_DIR, "..", "awkward-cpp", "tests-spec")
    if os.path.exists(tests_spec):
        shutil.rmtree(tests_spec)
    os.mkdir(tests_spec)
    with open(os.path.join(tests_spec, "__init__.py"), "w") as f:
        f.write(
            f"""# AUTO GENERATED ON {reproducible_datetime()}
# DO NOT EDIT BY HAND!
#
# To regenerate file, run
#
#     python dev/generate-tests.py
#

# fmt: off

"""
        )

    with open(
        os.path.join(CURRENT_DIR, "..", "awkward-cpp", "tests-spec", "kernels.py"), "w"
    ) as outfile:
        outfile.write(prefix)
        with open(
            os.path.join(CURRENT_DIR, "..", "kernel-specification.yml")
        ) as specfile:
            indspec = yaml.load(specfile, Loader=yaml.CSafeLoader)["kernels"]
            for spec in indspec:
                if "def " in spec["definition"]:
                    outfile.write(spec["definition"] + "\n")
                    for childfunc in spec["specializations"]:
                        outfile.write(
                            "{} = {}\n".format(childfunc["name"], spec["name"])
                        )
                    outfile.write("\n\n")

    unit_tests = os.path.join(CURRENT_DIR, "..", "awkward-cpp", "tests-spec-explicit")
    if os.path.exists(unit_tests):
        shutil.rmtree(unit_tests)
    os.mkdir(unit_tests)
    final_dest = os.path.join(CURRENT_DIR, "..", "awkward-cpp", "tests-spec-explicit")
    copy_dest = os.path.join(
        CURRENT_DIR, "..", "awkward-cpp", "tests-spec", "kernels.py"
    )
    shutil.copy(copy_dest, final_dest)


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


def genspectests(specdict):
    print("Generating files for testing specification")
    for spec in specdict.values():
        with open(
            os.path.join(
                CURRENT_DIR,
                "..",
                "awkward-cpp",
                "tests-spec",
                "test_py" + spec.name + ".py",
            ),
            "w",
        ) as f:
            f.write(
                f"""# AUTO GENERATED ON {reproducible_datetime()}
# DO NOT EDIT BY HAND!
#
# To regenerate file, run
#
#     python dev/generate-tests.py
#

# fmt: off

"""
            )

            f.write("import pytest\nimport numpy as np\nimport kernels\n\n")
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
                                    + f"assert {arg}[:len(pytest_{arg})] == pytest.approx(pytest_{arg})\n"
                                )
                            else:
                                f.write(" " * 4 + f"assert {arg} == pytest_{arg}\n")
                    else:
                        f.write(" " * 4 + "with pytest.raises(Exception):\n")
                        f.write(" " * 8 + "funcPy(" + args + ")\n")
                    f.write("\n")


def remove_const(typename):
    if "Const[" in typename:
        typename = typename.replace("Const[", "", 1).rstrip("]")
    return typename


def gencpukerneltests(specdict):
    print("Generating files for testing CPU kernels")

    tests_cpu_kernels = os.path.join(
        CURRENT_DIR, "..", "awkward-cpp", "tests-cpu-kernels"
    )
    if os.path.exists(tests_cpu_kernels):
        shutil.rmtree(tests_cpu_kernels)
    os.mkdir(tests_cpu_kernels)
    with open(os.path.join(tests_cpu_kernels, "__init__.py"), "w") as f:
        f.write(
            f"""# AUTO GENERATED ON {reproducible_datetime()}
# DO NOT EDIT BY HAND!
#
# To regenerate file, run
#
#     python dev/generate-tests.py
#

# fmt: off

"""
        )

    for spec in specdict.values():
        with open(
            os.path.join(tests_cpu_kernels, "test_cpu" + spec.name + ".py"), "w"
        ) as f:
            f.write(
                f"""# AUTO GENERATED ON {reproducible_datetime()}
# DO NOT EDIT BY HAND!
#
# To regenerate file, run
#
#     python dev/generate-tests.py
#

# fmt: off

"""
            )

            f.write(
                "import ctypes\n"
                "import numpy as np\n"
                "import pytest\n\n"
                "from awkward_cpp.cpu_kernels import lib\n\n"
            )
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
                    typename = gettype(arg, spec.args)
                    if "List" in typename:
                        count = typename.count("List")
                        typename = gettypename(typename)
                        if count == 1:
                            f.write(
                                " " * 4
                                + f"{arg} = (ctypes.c_{typename}*len({arg}))(*{arg})\n"
                            )
                        elif count == 2:
                            f.write(
                                " " * 4
                                + f"{typename}Ptr = ctypes.POINTER(ctypes.c_{typename})\n"
                                + " " * 4
                                + f"{typename}PtrPtr = ctypes.POINTER({typename}Ptr)\n"
                                + " " * 4
                                + f"dim0 = len({arg})\n"
                                + " " * 4
                                + f"dim1 = len({arg}[0])\n"
                                + " " * 4
                                + f"{arg}_np_arr_2d = np.empty([dim0, dim1], dtype=np.{typename})\n"
                                + " " * 4
                                + "for i in range(dim0):\n"
                                + " " * 8
                                + "for j in range(dim1):\n"
                                + " " * 12
                                + f"{arg}_np_arr_2d[i][j] = {arg}[i][j]\n"
                                + " " * 4
                                + f"{arg}_ct_arr = np.ctypeslib.as_ctypes({arg}_np_arr_2d)\n"
                                + " " * 4
                                + f"{typename}PtrArr = {typename}Ptr * {arg}_ct_arr._length_\n"
                                + " " * 4
                                + f"{arg}_ct_ptr = ctypes.cast({typename}PtrArr(*(ctypes.cast(row, {typename}Ptr) for row in {arg}_ct_arr)), {typename}PtrPtr)\n"
                                + " " * 4
                                + f"{arg} = {arg}_ct_ptr\n"
                            )
                f.write(" " * 4 + "funcC = getattr(lib, '" + spec.name + "')\n")
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
                            count = typename.count("List")
                            if count == 1:
                                f.write(
                                    " " * 4
                                    + f"assert {arg}[:len(pytest_{arg})] == pytest.approx(pytest_{arg})\n"
                                )
                            elif count == 2:
                                f.write(
                                    " " * 4
                                    + f"for row1, row2 in zip(pytest_{arg}, {arg}_np_arr_2d[:len(pytest_{arg})]):\n"
                                    + " " * 8
                                    + "assert row1 == pytest.approx(row2)\n"
                                )
                        else:
                            f.write(" " * 4 + f"assert {arg} == pytest_{arg}\n")
                    f.write(" " * 4 + "assert not ret_pass.str\n")
                else:
                    f.write(" " * 4 + f"assert funcC({args}).str\n")
                f.write("\n")


def gencpuunittests(specdict):
    print("Generating Unit Tests for CPU kernels")

    unit_test_map = unittestmap()
    unit_tests_cpu_kernels = os.path.join(
        CURRENT_DIR, "..", "awkward-cpp", "tests-cpu-kernels-explicit"
    )
    if os.path.exists(unit_tests_cpu_kernels):
        shutil.rmtree(unit_tests_cpu_kernels)
    os.mkdir(unit_tests_cpu_kernels)
    with open(os.path.join(unit_tests_cpu_kernels, "__init__.py"), "w") as f:
        f.write(
            f"""# AUTO GENERATED ON {reproducible_datetime()}
# DO NOT EDIT BY HAND!
#
# To regenerate file, run
#
#     python dev/generate-tests.py
#

# fmt: off

"""
        )

    for spec in specdict.values():
        if spec.templatized_kernel_name in list(unit_test_map.keys()):
            func = "test_unit_cpu" + spec.name + ".py"
            num = 1
            with open(os.path.join(unit_tests_cpu_kernels, func), "w") as f:
                f.write(
                    f"""# AUTO GENERATED ON {reproducible_datetime()}
# DO NOT EDIT BY HAND!
#
# To regenerate file, run
#
#     python dev/generate-tests.py
#

# fmt: off

"""
                )

                f.write(
                    "import ctypes\n"
                    "import numpy as np\n"
                    "import pytest\n\n"
                    "from awkward_cpp.cpu_kernels import lib\n\n"
                )
                unit_test_values = unit_test_map[spec.templatized_kernel_name]
                tests = unit_test_values["tests"]
                for test in tests:
                    funcName = (
                        "def test_unit_cpu" + spec.name + "_" + str(num) + "():\n"
                    )
                    unit_tests = getunittests(test["inputs"], test["outputs"])
                    flag = checkuint(unit_tests.items(), spec.args)
                    range = checkintrange(unit_tests.items(), test["error"], spec.args)
                    if flag and range:
                        num += 1
                        f.write(funcName)
                        for arg, val in test["outputs"].items():
                            dtype = gettype(arg, spec.args)
                            if "List" in dtype:
                                count = dtype.count("List")
                                typename = gettypename(dtype)
                                if count == 1:
                                    f.write(
                                        " " * 4
                                        + arg
                                        + " = "
                                        + str([gettypeval(dtype)] * len(val))
                                        + "\n"
                                    )
                                    f.write(
                                        " " * 4
                                        + f"{arg} = (ctypes.c_{typename}*len({arg}))(*{arg})\n"
                                    )
                                elif count == 2:
                                    f.write(" " * 4 + arg + " = [")
                                    for size, arr in enumerate(val):
                                        if size != 0:
                                            f.write(", ")
                                        f.write(str([gettypeval(dtype)] * len(arr)))
                                    f.write("]\n")
                                    f.write(
                                        " " * 4
                                        + f"{typename}Ptr = ctypes.POINTER(ctypes.c_{typename})\n"
                                        + " " * 4
                                        + f"{typename}PtrPtr = ctypes.POINTER({typename}Ptr)\n"
                                        + " " * 4
                                        + f"dim0 = len({arg})\n"
                                        + " " * 4
                                        + f"dim1 = len({arg}[0])\n"
                                        + " " * 4
                                        + f"{arg}_np_arr_2d = np.empty([dim0, dim1], dtype=np.{typename})\n"
                                        + " " * 4
                                        + "for i in range(dim0):\n"
                                        + " " * 8
                                        + "for j in range(dim1):\n"
                                        + " " * 12
                                        + f"{arg}_np_arr_2d[i][j] = {arg}[i][j]\n"
                                        + " " * 4
                                        + f"{arg}_ct_arr = np.ctypeslib.as_ctypes({arg}_np_arr_2d)\n"
                                        + " " * 4
                                        + f"{typename}PtrArr = {typename}Ptr * {arg}_ct_arr._length_\n"
                                        + " " * 4
                                        + f"{arg}_ct_ptr = ctypes.cast({typename}PtrArr(*(ctypes.cast(row, {typename}Ptr) for row in {arg}_ct_arr)), {typename}PtrPtr)\n"
                                        + " " * 4
                                        + f"{arg} = {arg}_ct_ptr\n"
                                    )
                            else:
                                f.write(
                                    " " * 4
                                    + arg
                                    + " = "
                                    + str([gettypeval(dtype)] * len(val))
                                    + "\n"
                                )
                        for arg, val in test["inputs"].items():
                            typename = gettype(arg, spec.args)
                            f.write(" " * 4 + arg + " = " + str(val) + "\n")
                            if "List" in typename:
                                count = typename.count("List")
                                typename = gettypename(typename)
                                if count == 1:
                                    f.write(
                                        " " * 4
                                        + f"{arg} = (ctypes.c_{typename}*len({arg}))(*{arg})\n"
                                    )
                                elif count == 2:
                                    f.write(
                                        " " * 4
                                        + f"{typename}Ptr = ctypes.POINTER(ctypes.c_{typename})\n"
                                        + " " * 4
                                        + f"{typename}PtrPtr = ctypes.POINTER({typename}Ptr)\n"
                                        + " " * 4
                                        + f"dim0 = len({arg})\n"
                                        + " " * 4
                                        + f"dim1 = len({arg}[0])\n"
                                        + " " * 4
                                        + f"{arg}_np_arr_2d = np.empty([dim0, dim1], dtype=np.{typename})\n"
                                        + " " * 4
                                        + "for i in range(dim0):\n"
                                        + " " * 8
                                        + "for j in range(dim1):\n"
                                        + " " * 12
                                        + f"{arg}_np_arr_2d[i][j] = {arg}[i][j]\n"
                                        + " " * 4
                                        + f"{arg}_ct_arr = np.ctypeslib.as_ctypes({arg}_np_arr_2d)\n"
                                        + " " * 4
                                        + f"{typename}PtrArr = {typename}Ptr * {arg}_ct_arr._length_\n"
                                        + " " * 4
                                        + f"{arg}_ct_ptr = ctypes.cast({typename}PtrArr(*(ctypes.cast(row, {typename}Ptr) for row in {arg}_ct_arr)), {typename}PtrPtr)\n"
                                        + " " * 4
                                        + f"{arg} = {arg}_ct_ptr\n"
                                    )

                        f.write(" " * 4 + "funcC = getattr(lib, '" + spec.name + "')\n")
                        args = ""
                        count = 0
                        for arg in spec.args:
                            if count == 0:
                                args += arg.name
                                count += 1
                            else:
                                args += ", " + arg.name
                        if not test["error"]:
                            f.write(" " * 4 + "ret_pass = funcC(" + args + ")\n")
                            for arg, val in test["outputs"].items():
                                typename = gettype(arg, spec.args)
                                if "bool" in typename:
                                    output = []
                                    if "List[List" in typename:
                                        for row in val:
                                            for data in row:
                                                if data >= 1:
                                                    data = 1
                                                output.append(data)
                                        val = output
                                    elif "List" in typename:
                                        for data in val:
                                            if data >= 1:
                                                data = 1
                                            output.append(data)
                                        val = output
                                    else:
                                        if val >= 1:
                                            val = 1

                                f.write(
                                    " " * 4 + "pytest_" + arg + " = " + str(val) + "\n"
                                )
                                if isinstance(val, list):
                                    count = typename.count("List")
                                    if count == 1:
                                        f.write(
                                            " " * 4
                                            + f"assert {arg}[:len(pytest_{arg})] == pytest.approx(pytest_{arg})\n"
                                        )
                                    elif count == 2:
                                        f.write(
                                            " " * 4
                                            + f"for row1, row2 in zip(pytest_{arg}, {arg}_np_arr_2d[:len(pytest_{arg})]):\n"
                                            + " " * 8
                                            + "assert row1 == pytest.approx(row2)\n"
                                        )
                                else:
                                    f.write(" " * 4 + f"assert {arg} == pytest_{arg}\n")
                            f.write(" " * 4 + "assert not ret_pass.str\n")
                        else:
                            f.write(
                                " " * 4
                                + f"assert funcC({args}).str.decode('utf-8') == \"{test['message']}\"\n"
                            )
                        f.write("\n")


cuda_kernels_tests = [
    "awkward_Index_nones_as_index",
    "awkward_ListArray_min_range",
    "awkward_ListArray_validity",
    "awkward_BitMaskedArray_to_ByteMaskedArray",
    "awkward_ListArray_broadcast_tooffsets",
    "awkward_ListArray_compact_offsets",
    "awkward_ListOffsetArray_flatten_offsets",
    "awkward_IndexedArray_overlay_mask",
    "awkward_ByteMaskedArray_numnull",
    "awkward_IndexedArray_numnull",
    "awkward_IndexedArray_numnull_parents",
    "awkward_IndexedArray_numnull_unique_64",
    "awkward_ListArray_fill",
    "awkward_IndexedArray_fill",
    "awkward_IndexedArray_fill_count",
    "awkward_UnionArray_fillindex",
    "awkward_UnionArray_fillindex_count",
    "awkward_UnionArray_fillna",
    "awkward_UnionArray_filltags",
    "awkward_UnionArray_filltags_const",
    "awkward_localindex",
    "awkward_IndexedArray_reduce_next_fix_offsets_64",
    "awkward_RegularArray_getitem_next_array_advanced",
    "awkward_ByteMaskedArray_toIndexedOptionArray",
    "awkward_IndexedArray_simplify",
    "awkward_UnionArray_validity",
    "awkward_IndexedArray_validity",
    "awkward_ByteMaskedArray_overlay_mask",
    "awkward_NumpyArray_reduce_mask_ByteMaskedArray_64",
    "awkward_RegularArray_getitem_carry",
    "awkward_RegularArray_localindex",
    "awkward_RegularArray_rpad_and_clip_axis1",
    "awkward_RegularArray_getitem_next_range",
    "awkward_RegularArray_getitem_next_range_spreadadvanced",
    "awkward_RegularArray_getitem_next_array",
    "awkward_RegularArray_getitem_next_array_regularize",
    "awkward_RegularArray_reduce_local_nextparents_64",
    "awkward_RegularArray_reduce_nonlocal_preparenext_64",
    "awkward_missing_repeat",
    "awkward_RegularArray_getitem_jagged_expand",
    "awkward_ListArray_combinations_length",
    "awkward_ListArray_combinations",
    "awkward_RegularArray_combinations_64",
    "awkward_ListArray_getitem_jagged_apply",
    "awkward_ListArray_getitem_jagged_carrylen",
    "awkward_ListArray_getitem_jagged_descend",
    "awkward_ListArray_getitem_jagged_expand",
    "awkward_ListArray_getitem_jagged_numvalid",
    "awkward_ListArray_getitem_jagged_shrink",
    "awkward_ListArray_getitem_next_array_advanced",
    "awkward_ListArray_getitem_next_array",
    "awkward_ListArray_getitem_next_at",
    "awkward_ListArray_getitem_next_range",
    "awkward_ListArray_getitem_next_range_carrylength",
    "awkward_ListArray_getitem_next_range_counts",
    "awkward_ListArray_rpad_and_clip_length_axis1",
    "awkward_ListArray_rpad_axis1",
    "awkward_UnionArray_regular_index",
    "awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64",
    "awkward_ListArray_getitem_next_range_spreadadvanced",
    "awkward_ListArray_localindex",
    "awkward_NumpyArray_pad_zero_to_length",
    "awkward_NumpyArray_reduce_adjust_starts_64",
    "awkward_NumpyArray_rearrange_shifted",
    "awkward_NumpyArray_reduce_adjust_starts_shifts_64",
    "awkward_RegularArray_getitem_next_at",
    "awkward_BitMaskedArray_to_IndexedOptionArray",
    "awkward_ByteMaskedArray_getitem_nextcarry",
    "awkward_ByteMaskedArray_getitem_nextcarry_outindex",
    "awkward_ByteMaskedArray_reduce_next_64",
    "awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64",
    "awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_fromshifts_64",
    "awkward_Content_getitem_next_missing_jagged_getmaskstartstop",
    "awkward_index_rpad_and_clip_axis0",
    "awkward_index_rpad_and_clip_axis1",
    "awkward_NumpyArray_subrange_equal",
    "awkward_NumpyArray_subrange_equal_bool",
    "awkward_IndexedArray_flatten_nextcarry",
    "awkward_IndexedArray_flatten_none2empty",
    "awkward_IndexedArray_getitem_nextcarry",
    "awkward_IndexedArray_getitem_nextcarry_outindex",
    "awkward_IndexedArray_index_of_nulls",
    "awkward_IndexedArray_local_preparenext_64",
    "awkward_IndexedArray_ranges_next_64",
    "awkward_IndexedArray_ranges_carry_next_64",
    "awkward_IndexedArray_reduce_next_64",
    "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64",
    "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64",
    "awkward_IndexedOptionArray_rpad_and_clip_mask_axis1",
    "awkward_ListOffsetArray_local_preparenext_64",
    "awkward_ListOffsetArray_rpad_and_clip_axis1",
    "awkward_ListOffsetArray_rpad_length_axis1",
    "awkward_ListOffsetArray_toRegularArray",
    "awkward_ListOffsetArray_rpad_axis1",
    "awkward_MaskedArray_getitem_next_jagged_project",
    "awkward_UnionArray_project",
    "awkward_ListOffsetArray_drop_none_indexes",
    "awkward_ListOffsetArray_reduce_local_nextparents_64",
    "awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64",
    "awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64",
    "awkward_ListOffsetArray_reduce_local_outoffsets_64",
    "awkward_UnionArray_flatten_length",
    "awkward_UnionArray_flatten_combine",
    "awkward_UnionArray_nestedfill_tags_index",
    "awkward_UnionArray_regular_index_getsize",
    "awkward_UnionArray_simplify",
    "awkward_UnionArray_simplify_one",
    "awkward_RecordArray_reduce_nonlocal_outoffsets_64",
    "awkward_reduce_count_64",
    "awkward_reduce_max",
    "awkward_reduce_max_complex",
    "awkward_reduce_min",
    "awkward_reduce_min_complex",
    "awkward_reduce_sum",
    "awkward_reduce_sum_bool",
    "awkward_reduce_sum_bool_complex",
    "awkward_reduce_sum_complex",
    "awkward_reduce_sum_int32_bool_64",
    "awkward_reduce_sum_int64_bool_64",
    "awkward_reduce_prod",
    "awkward_reduce_prod_bool",
    "awkward_reduce_prod_bool_complex",
    "awkward_reduce_prod_complex",
    "awkward_reduce_countnonzero",
    "awkward_reduce_countnonzero_complex",
    "awkward_sorting_ranges",
    "awkward_sorting_ranges_length",
]


def gencudakerneltests(specdict):
    print("Generating files for testing CUDA kernels")

    tests_cuda_kernels = os.path.join(CURRENT_DIR, "..", "tests-cuda-kernels")
    if os.path.exists(tests_cuda_kernels):
        shutil.rmtree(tests_cuda_kernels)
    os.mkdir(tests_cuda_kernels)
    with open(os.path.join(tests_cuda_kernels, "__init__.py"), "w") as f:
        f.write(
            f"""# AUTO GENERATED ON {reproducible_datetime()}
# DO NOT EDIT BY HAND!
#
# To regenerate file, run
#
#     python dev/generate-tests.py
#

# fmt: off

"""
        )

    for spec in specdict.values():
        if spec.templatized_kernel_name in cuda_kernels_tests:
            with open(
                os.path.join(tests_cuda_kernels, "test_cuda" + spec.name + ".py"), "w"
            ) as f:
                f.write(
                    f"""# AUTO GENERATED ON {reproducible_datetime()}
# DO NOT EDIT BY HAND!
#
# To regenerate file, run
#
#     python dev/generate-tests.py
#

# fmt: off

"""
                )

                f.write(
                    "import cupy\n"
                    "import cupy.testing as cpt\n"
                    "import numpy as np\n"
                    "import pytest\n\n"
                    "import awkward as ak\n"
                    "import awkward._connect.cuda as ak_cu\n"
                    "from awkward._backends.cupy import CupyBackend\n\n"
                    "cupy_backend = CupyBackend.instance()\n\n"
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
                    dtypes = []
                    for arg, val in test["inargs"].items():
                        typename = gettype(arg, spec.args)
                        if "List" not in typename:
                            f.write(" " * 4 + arg + " = " + str(val) + "\n")
                        if "List" in typename:
                            count = typename.count("List")
                            typename = gettypename(typename)
                            if typename == "bool":
                                typename = typename + "_"
                            if typename == "float":
                                typename = typename + "32"
                            if count == 1:
                                f.write(
                                    " " * 4
                                    + f"{arg} = cupy.array({val}, dtype=cupy.{typename})\n"
                                )
                                dtypes.append("cupy." + typename)
                            elif count == 2:
                                raise NotImplementedError
                    cuda_string = (
                        "funcC = cupy_backend['"
                        + spec.templatized_kernel_name
                        + "', {}]\n".format(", ".join(dtypes))
                    )
                    f.write(" " * 4 + cuda_string)
                    args = ""
                    count = 0
                    for arg in spec.args:
                        if count == 0:
                            args += arg.name
                            count += 1
                        else:
                            args += ", " + arg.name
                    if test["success"]:
                        f.write(" " * 4 + "funcC(" + args + ")\n")
                        f.write(
                            """
    try:
        ak_cu.synchronize_cuda()
    except:
        pytest.fail("This test case shouldn't have raised an error")
"""
                        )

                        for arg, val in test["outargs"].items():
                            f.write(" " * 4 + "pytest_" + arg + " = " + str(val) + "\n")
                            if isinstance(val, list):
                                f.write(
                                    " " * 4
                                    + f"cpt.assert_allclose({arg}[:len(pytest_{arg})], cupy.array(pytest_{arg}))\n"
                                )
                            else:
                                f.write(" " * 4 + f"assert {arg} == pytest_{arg}\n")
                    f.write("\n")


def gencudaunittests(specdict):
    print("Generating Unit Tests for CUDA kernels")

    unit_test_map = unittestmap()
    unit_tests_cuda_kernels = os.path.join(
        CURRENT_DIR, "..", "tests-cuda-kernels-explicit"
    )
    if os.path.exists(unit_tests_cuda_kernels):
        shutil.rmtree(unit_tests_cuda_kernels)
    os.mkdir(unit_tests_cuda_kernels)
    with open(os.path.join(unit_tests_cuda_kernels, "__init__.py"), "w") as f:
        f.write(
            f"""# AUTO GENERATED ON {reproducible_datetime()}
# DO NOT EDIT BY HAND!
#
# To regenerate file, run
#
#     python dev/generate-tests.py
#

# fmt: off

"""
        )

    for spec in specdict.values():
        if (
            spec.templatized_kernel_name in cuda_kernels_tests
            and spec.templatized_kernel_name in list(unit_test_map.keys())
        ):
            func = "test_unit_cuda" + spec.name + ".py"
            num = 1
            with open(
                os.path.join(unit_tests_cuda_kernels, func),
                "w",
            ) as f:
                f.write(
                    f"""# AUTO GENERATED ON {reproducible_datetime()}
# DO NOT EDIT BY HAND!
#
# To regenerate file, run
#
#     python dev/generate-tests.py
#

# fmt: off

"""
                )

                f.write(
                    "import re\n"
                    "import cupy\n"
                    "import cupy.testing as cpt\n"
                    "import pytest\n\n"
                    "import awkward as ak\n"
                    "import awkward._connect.cuda as ak_cu\n"
                    "from awkward._backends.cupy import CupyBackend\n\n"
                    "cupy_backend = CupyBackend.instance()\n\n"
                )
                unit_test_values = unit_test_map[spec.templatized_kernel_name]
                tests = unit_test_values["tests"]
                status = unit_test_values["status"]
                for test in tests:
                    funcName = (
                        "def test_unit_cuda" + spec.name + "_" + str(num) + "():\n"
                    )
                    dtypes = getdtypes(spec.args)
                    unit_tests = getunittests(test["inputs"], test["outputs"])
                    flag = checkuint(unit_tests.items(), spec.args)
                    range = checkintrange(unit_tests.items(), test["error"], spec.args)
                    if flag and range:
                        num += 1
                        if not status:
                            f.write(
                                "@pytest.mark.skip(reason='Kernel is not implemented properly')\n"
                            )
                        f.write(funcName)
                        for arg, val in test["outputs"].items():
                            typename = gettype(arg, spec.args)
                            if "List" not in typename:
                                f.write(" " * 4 + arg + " = " + str(val) + "\n")
                            if "List" in typename:
                                count = typename.count("List")
                                typename = gettypename(typename)
                                if typename == "bool":
                                    typename = typename + "_"
                                if typename == "float":
                                    typename = typename + "32"
                                if count == 1:
                                    f.write(
                                        " " * 4
                                        + f"{arg} = cupy.array({[gettypeval(typename)] * len(val)}, dtype=cupy.{typename})\n"
                                    )
                                elif count == 2:
                                    f.write(
                                        " " * 4
                                        + f"{arg} = cupy.array({val}, dtype=cupy.{typename})\n"
                                        + " " * 4
                                        + f"{arg}_array = [cupy.array(row, dtype=cupy.{typename}) for row in {arg}]\n"
                                        + " " * 4
                                        + f"{arg} = cupy.array([row.data.ptr for row in {arg}_array])\n"
                                    )
                        for arg, val in test["inputs"].items():
                            typename = gettype(arg, spec.args)
                            if "List" not in typename:
                                f.write(" " * 4 + arg + " = " + str(val) + "\n")
                            if "List" in typename:
                                count = typename.count("List")
                                typename = gettypename(typename)
                                if typename == "bool":
                                    typename = typename + "_"
                                if typename == "float":
                                    typename = typename + "32"
                                if count == 1:
                                    f.write(
                                        " " * 4
                                        + f"{arg} = cupy.array({val}, dtype=cupy.{typename})\n"
                                    )
                                elif count == 2:
                                    f.write(
                                        " " * 4
                                        + f"{arg} = cupy.array({val}, dtype=cupy.{typename})\n"
                                        + " " * 4
                                        + f"{arg}_array = [cupy.array(row, dtype=cupy.{typename}) for row in {arg}]\n"
                                        + " " * 4
                                        + f"{arg} = cupy.array([row.data.ptr for row in {arg}_array])\n"
                                    )
                        cuda_string = (
                            "funcC = cupy_backend['"
                            + spec.templatized_kernel_name
                            + "', {}]\n".format(", ".join(dtypes))
                        )
                        f.write(" " * 4 + cuda_string)
                        args = ""
                        count = 0
                        for arg in spec.args:
                            if count == 0:
                                args += arg.name
                                count += 1
                            else:
                                args += ", " + arg.name
                        f.write(" " * 4 + "funcC(" + args + ")\n")
                        if test["error"]:
                            f.write(
                                f"""
    error_message = re.escape("{test['message']} in compiled CUDA code ({spec.templatized_kernel_name})")
"""
                            )
                            f.write(
                                """    with pytest.raises(ValueError, match=rf"{error_message}"):
        ak_cu.synchronize_cuda()
"""
                            )
                        else:
                            f.write(
                                """
    try:
        ak_cu.synchronize_cuda()
    except:
        pytest.fail("This test case shouldn't have raised an error")
"""
                            )
                            for arg, val in test["outputs"].items():
                                typename = gettype(arg, spec.args)
                                if "bool" in typename:
                                    output = []
                                    if "List[List" in typename:
                                        for row in val:
                                            for data in row:
                                                if data >= 1:
                                                    data = 1
                                                output.append(data)
                                        val = output
                                    elif "List" in typename:
                                        for data in val:
                                            if data >= 1:
                                                data = 1
                                            output.append(data)
                                        val = output
                                    else:
                                        if val >= 1:
                                            val = 1

                                f.write(
                                    " " * 4 + "pytest_" + arg + " = " + str(val) + "\n"
                                )
                                if isinstance(val, list):
                                    count = typename.count("List")
                                    if count == 1:
                                        f.write(
                                            " " * 4
                                            + f"cpt.assert_allclose({arg}[:len(pytest_{arg})], cupy.array(pytest_{arg}))\n"
                                        )
                                    elif count == 2:
                                        f.write(
                                            " " * 4
                                            + f"for row1, row2 in zip(pytest_{arg}, {arg}_array[:len(pytest_{arg})]):\n"
                                            + " " * 8
                                            + "cpt.assert_allclose(row1, row2)\n"
                                        )
                                else:
                                    f.write(" " * 4 + f"assert {arg} == pytest_{arg}\n")
                        f.write("\n")


def genunittests():
    print("Generating Unit Tests")
    with open(os.path.join(CURRENT_DIR, "..", "kernel-test-data.json")) as f:
        data = json.load(f)["unit-tests"]

    for function in data:
        num = 0
        func = "test_" + function["name"] + ".py"
        with open(
            os.path.join(CURRENT_DIR, "..", "awkward-cpp", "tests-spec-explicit", func),
            "w",
        ) as file:
            file.write("import pytest\n" "import numpy\n" "import kernels\n\n")
            for test in function["tests"]:
                num += 1
                funcName = "def test_" + function["name"] + "_" + str(num) + "():\n"
                file.write(funcName)
                for key, value in test["outputs"].items():
                    if all(isinstance(elem, list) for elem in value) and value:
                        file.write("\t" + key + " = [")
                        for size in range(len(value)):
                            if size != 0:
                                file.write(", ")
                            file.write(str([123] * len(value[0])))
                        file.write("]\n")
                    else:
                        file.write("\t" + key + " = " + str([123] * len(value)) + "\n")
                for key, value in test["inputs"].items():
                    file.write("\t" + key + " = " + str(value) + "\n")
                file.write("\tfuncPy = getattr(kernels, '" + function["name"] + "')\n")
                line = "\tfuncPy("
                for key in test["outputs"]:
                    line += key + " = " + key + ","
                for key in test["inputs"]:
                    if key not in test["outputs"]:
                        line += key + " = " + key + ","
                line = line[0 : len(line) - 1]
                line += ")\n"
                if test["error"]:
                    file.write("\twith pytest.raises(Exception):\n")
                    file.write("\t" + line)
                else:
                    file.write(line)
                    for key, value in test["outputs"].items():
                        file.write("\tpytest_" + key + " = " + str(value) + "\n")
                    for key in test["outputs"]:
                        file.write("\tassert " + key + " == " + "pytest_" + key + "\n")
                file.write("\n\n")


def evalkernels():
    with open(
        os.path.join(CURRENT_DIR, "..", "awkward-cpp", "tests-spec", "kernels.py")
    ) as kernelfile:
        exec(kernelfile.read(), globals())


if __name__ == "__main__":
    genpykernels()
    evalkernels()
    specdict, specdict_unit = readspec()
    genspectests(specdict)
    gencpukerneltests(specdict)
    gencpuunittests(specdict_unit)
    genunittests()
    gencudakerneltests(specdict)
    gencudaunittests(specdict_unit)
