# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import copy
import datetime
import json
import os
import shutil
import time
from collections import OrderedDict
from itertools import product

import yaml
from numpy import uint8  # noqa: F401 (used in evaluated strings)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def reproducible_datetime():
    build_date = datetime.datetime.utcfromtimestamp(
        int(os.environ.get("SOURCE_DATE_EPOCH", time.time()))
    )
    return build_date.isoformat().replace("T", " AT ")[:22]


class Argument:
    __slots__ = ("name", "typename", "direction", "role")

    def __init__(self, name, typename, direction, role="default"):
        self.name = name
        self.typename = typename
        self.direction = direction
        self.role = role


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
            funcPy = eval(self.name, globals(), locals())  # noqa: PGH001

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
    with open(os.path.join(CURRENT_DIR, "..", "kernel-specification.yml")) as f:
        loadfile = yaml.load(f, Loader=yaml.CSafeLoader)

    indspec = loadfile["kernels"]
    with open(os.path.join(CURRENT_DIR, "..", "kernel-test-data.json")) as f:
        data = json.load(f)["tests"]

    for spec in indspec:
        if "def " in spec["definition"]:
            for childfunc in spec["specializations"]:
                specdict[childfunc["name"]] = Specification(
                    spec["name"],
                    childfunc,
                    data,
                    not spec["automatic-tests"],
                )
    return specdict


def gettypename(spectype):
    typename = spectype.replace("List", "").replace("[", "").replace("]", "")
    if typename.endswith("_t"):
        typename = typename[:-2]
    return typename


def getfuncnames():
    funcs = {}
    with open(os.path.join(CURRENT_DIR, "..", "kernel-specification.yml")) as specfile:
        indspec = yaml.load(specfile, Loader=yaml.CSafeLoader)["kernels"]
        for spec in indspec:
            funcs[spec["name"]] = []
            for childfunc in spec["specializations"]:
                funcs[spec["name"]].append(childfunc["name"])
    return funcs


def genpykernels():
    print("Generating Python kernels")
    prefix = """
from numpy import uint8
kMaxInt64  = 9223372036854775806
kSliceNone = kMaxInt64 + 1
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


def getcudakernelslist():
    cudakernels = []
    for f in os.listdir(
        os.path.join(
            os.path.dirname(CURRENT_DIR),
            "src",
            "awkward",
            "_connect",
            "cuda",
            "cuda_kernels",
        )
    ):
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
                "import ctypes\nimport pytest\n\nfrom awkward_cpp.cpu_kernels import lib\n\n"
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
                                + f"{arg} = (ctypes.c_{typename}*len({arg}))(*{arg})\n"
                            )
                        elif count == 2:
                            f.write(
                                " " * 4
                                + "{0} = ctypes.pointer(ctypes.cast((ctypes.c_{1}*len({0}[0]))(*{0}[0]),ctypes.POINTER(ctypes.c_{1})))\n".format(
                                    arg, typename
                                )
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
                            f.write(
                                " " * 4
                                + f"assert {arg}[:len(pytest_{arg})] == pytest.approx(pytest_{arg})\n"
                            )
                        else:
                            f.write(" " * 4 + f"assert {arg} == pytest_{arg}\n")
                    f.write(" " * 4 + "assert not ret_pass.str\n")
                else:
                    f.write(" " * 4 + f"assert funcC({args}).str\n")
                f.write("\n")


cuda_kernels_tests = [
    "awkward_ListArray_validity",
    "awkward_BitMaskedArray_to_ByteMaskedArray",
    "awkward_ListArray_compact_offsets",
    "awkward_new_Identities",
    "awkward_Identities32_to_Identities64",
    "awkward_ListOffsetArray_flatten_offsets",
    "awkward_IndexedArray_overlay_mask",
    "awkward_IndexedArray_mask",
    "awkward_ByteMaskedArray_mask",
    "awkward_zero_mask",
    "awkward_IndexedArray_fill_count",
    "awkward_UnionArray_fillna",
    "awkward_localindex",
    "awkward_content_reduce_zeroparents_64",
    "awkward_ListOffsetArray_reduce_global_startstop_64",
    "awkward_IndexedArray_reduce_next_fix_offsets_64",
    "awkward_Index_to_Index64",
    "awkward_carry_arange",
    "awkward_index_carry_nocheck",
    "awkward_NumpyArray_contiguous_init",
    "awkward_NumpyArray_getitem_next_array_advanced",
    "awkward_NumpyArray_getitem_next_at",
    "awkward_RegularArray_getitem_next_array_advanced",
    "awkward_ByteMaskedArray_toIndexedOptionArray",
    "awkward_combinations",  # ?
    "awkward_IndexedArray_simplify",
    "awkward_UnionArray_validity",
    "awkward_index_carry",
    "awkward_ByteMaskedArray_getitem_carry",
    "awkward_IndexedArray_validity",
    "awkward_ByteMaskedArray_overlay_mask",
    "awkward_NumpyArray_reduce_mask_ByteMaskedArray_64",
    "awkward_RegularArray_getitem_carry",
    "awkward_NumpyArray_getitem_next_array",
    "awkward_RegularArray_localindex",
    "awkward_NumpyArray_contiguous_next",
    "awkward_NumpyArray_getitem_next_range",
    "awkward_NumpyArray_getitem_next_range_advanced",
    "awkward_RegularArray_getitem_next_range",
    "awkward_RegularArray_getitem_next_range_spreadadvanced",
    "awkward_RegularArray_getitem_next_array",
    "awkward_missing_repeat",
    "awkward_Identities_getitem_carry",
    "awkward_RegularArray_getitem_jagged_expand",
    "awkward_ListArray_getitem_jagged_expand",
    "awkward_ListArray_getitem_next_array",
    "awkward_NumpyArray_fill_tobool",
    "awkward_NumpyArray_reduce_adjust_starts_64",
    "awkward_NumpyArray_reduce_adjust_starts_shifts_64",
    "awkward_regularize_arrayslice",
    "awkward_RegularArray_getitem_next_at",
    "awkward_BitMaskedArray_to_IndexedOptionArray",
    "awkward_ByteMaskedArray_getitem_nextcarry",
    "awkward_ByteMaskedArray_getitem_nextcarry_outindex",
    "awkward_ByteMaskedArray_reduce_next_64",
    "awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64",
    "awkward_Content_getitem_next_missing_jagged_getmaskstartstop",
    "awkward_index_rpad_and_clip_axis1",
    "awkward_IndexedArray_flatten_nextcarry",
    "awkward_IndexedArray_getitem_nextcarry",
    "awkward_IndexedArray_getitem_nextcarry_outindex",
    "awkward_IndexedArray_getitem_nextcarry_outindex_mask",
    "awkward_IndexedArray_reduce_next_64",
    "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64",
    "awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64",
    "awkward_IndexedOptionArray_rpad_and_clip_mask_axis1",
    "awkward_ListOffsetArray_rpad_and_clip_axis1",
    # "awkward_ListOffsetArray_rpad_axis1",
    "awkward_MaskedArray_getitem_next_jagged_project",
    "awkward_NumpyArray_getitem_boolean_nonzero",
    "awkward_UnionArray_project",
    "awkward_reduce_argmax",
    "awkward_reduce_argmax_bool_64",
    "awkward_reduce_argmin",
    "awkward_reduce_argmin_bool_64",
    "awkward_reduce_count_64",
    "awkward_reduce_max",
    "awkward_reduce_min",
    "awkward_reduce_sum",
    "awkward_reduce_sum_int32_bool_64",
    "awkward_reduce_sum_int64_bool_64",
    "awkward_reduce_sum_bool",
    "awkward_reduce_prod_bool",
    "awkward_reduce_countnonzero",
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
                    "import cupy\nimport pytest\n\nimport awkward as ak\nimport awkward._connect.cuda as ak_cu\n\ncupy_backend = ak._backends.CupyBackend.instance()\n\n"
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
                    f.write("def test_cuda_" + spec.name + "_" + str(num) + "():\n")
                    num += 1
                    dtypes = []
                    for arg, val in test["inargs"].items():
                        typename = remove_const(
                            next(
                                argument
                                for argument in spec.args
                                if argument.name == arg
                            ).typename
                        )
                        if "List" not in typename:
                            f.write(" " * 4 + arg + " = " + str(val) + "\n")
                        if "List" in typename:
                            count = typename.count("List")
                            typename = gettypename(typename)
                            if typename == "bool" or typename == "float":
                                typename = typename + "_"
                            if count == 1:
                                f.write(
                                    " " * 4
                                    + "{} = cupy.array({}, dtype=cupy.{})\n".format(
                                        arg, val, typename
                                    )
                                )
                                dtypes.append("cupy." + typename)
                            elif count == 2:
                                raise NotImplementedError
                                # f.write(
                                #     " " * 4
                                #     + "{0} = ctypes.pointer(ctypes.cast((ctypes.c_{1}*len({0}[0]))(*{0}[0]),ctypes.POINTER(ctypes.c_{1})))\n".format(
                                #         arg, typename
                                #     )
                                # )
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
                                    + f"assert cupy.array_equal({arg}[:len(pytest_{arg})], cupy.array(pytest_{arg}))\n"
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
            file.write("import pytest\nimport kernels\n\n")
            for test in function["tests"]:
                num += 1
                funcName = "def test_" + function["name"] + "_" + str(num) + "():\n"
                file.write(funcName)
                for key, value in test["outputs"].items():
                    file.write("\t" + key + " = " + str([123] * len(value)) + "\n")
                for key, value in test["inputs"].items():
                    file.write("\t" + key + " = " + str(value) + "\n")
                file.write("\tfuncPy = getattr(kernels, '" + function["name"] + "')\n")
                line = "\tfuncPy("
                for key in test["outputs"]:
                    line += key + " = " + key + ","
                for key in test["inputs"]:
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
    specdict = readspec()
    genspectests(specdict)
    gencpukerneltests(specdict)
    genunittests()
    gencudakerneltests(specdict)
