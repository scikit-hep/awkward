# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import ctypes
import json
import math
import os
import re

from parser_utils import gettokens

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
for root, _, files in os.walk(CURRENT_DIR[:-3]):
    for filename in files:
        if filename.endswith("libawkward-cpu-kernels.so"):
            CPU_KERNEL_SO = os.path.join(root, filename)
            break


class Error(ctypes.Structure):
    _fields_ = [
        ("str", ctypes.POINTER(ctypes.c_char)),
        ("identity", ctypes.c_int64),
        ("attempt", ctypes.c_int64),
        ("pass_through", ctypes.c_bool),
    ]


def gentests(funcs, htokens, failfuncs):
    import kernels

    print("Executing tests")

    def pytype(cpptype):
        if re.match("u?int\d{1,2}(_t)?", cpptype) is not None:
            return "int"
        elif cpptype == "double":
            return "float"
        else:
            return cpptype

    def getctypelist(typelist):
        newctypes = []
        for x in typelist:
            if isinstance(x, tuple) and len(x) == 2:
                mid = x[0]
                if mid.endswith("_t"):
                    mid = mid[:-2]
                start = ""
                end = ")"
                for i in range(x[1]):
                    if i > 0:
                        start += "("
                        end += ")"
                    start += "ctypes.POINTER"
                start += "(ctypes.c_"
                newctypes.append(eval(start + mid + end))
            else:
                if x.endswith("_t"):
                    x = x[:-2]
                newctypes.append(eval("ctypes.c_" + x))
        return tuple(newctypes)

    tokens = gettokens(funcs, htokens)

    lib = ctypes.CDLL(CPU_KERNEL_SO)

    with open(os.path.join(CURRENT_DIR, "testcases.json")) as f:
        data = json.load(f)
        blacklist = [
            "awkward_ListArray32_combinations_64",
            "awkward_ListArrayU32_combinations_64",
            "awkward_ListArray64_combinations_64",
            "awkward_RegularArray_combinations_64",
        ]
        success_blacklist = ["awkward_combinations_64"]
        failure_blacklist = ["awkward_NumpyArray_fill_to64_fromU64"]
        for name, args in tokens.items():
            checkindex = []
            pindex = []
            pval = []
            typelist = []
            testsp = []
            testsc = []
            argnames = []
            if name not in blacklist:
                for i in range(len(args.values())):
                    argnames.append(list(args.keys())[i])
                    if list(args.values())[i]["type"].endswith("_t"):
                        temptype = list(args.values())[i]["type"][:-2]
                    elif list(args.values())[i]["type"] == "int":
                        temptype = "int64"
                    else:
                        temptype = list(args.values())[i]["type"]
                    if list(args.values())[i]["check"] == "outparam":
                        typelist.append((temptype, list(args.values())[i]["array"],))
                        temparr = [0] * (data["success"]["num"] + 50)
                        testsp.append(temparr)
                        if (
                            "role" in list(args.values())[i]
                            and list(args.values())[i]["role"] == "pointer"
                        ):
                            pval.append(eval("ctypes.c_" + temptype + "(0)"))
                            testsc.append(ctypes.byref(pval[-1]))
                            pindex.append(i)
                        else:
                            testsc.append(
                                (eval("ctypes.c_" + temptype) * len(temparr))(*temparr)
                            )
                            checkindex.append(i)
                    elif "role" in list(args.values())[i]:
                        if "instance" in list(args.values())[i]:
                            tempval = data["success"][
                                list(args.values())[i]["role"][
                                    : list(args.values())[i]["role"].find("-")
                                ]
                            ][list(args.values())[i]["role"]][
                                int(list(args.values())[i]["instance"])
                            ][
                                pytype(temptype)
                            ]
                        else:
                            tempval = data["success"][
                                list(args.values())[i]["role"][
                                    : list(args.values())[i]["role"].find("-")
                                ]
                            ][list(args.values())[i]["role"]][0][pytype(temptype)]
                        testsp.append(tempval)
                        if not isinstance(tempval, list):
                            typelist.append((temptype))
                            testsc.append(tempval)
                        else:
                            typelist.append(
                                (temptype, list(args.values())[i]["array"],)
                            )
                            if list(args.values())[i]["array"] == 2:
                                testsc.append(
                                    (
                                        eval(
                                            "ctypes.pointer(ctypes.cast((ctypes.c_"
                                            + temptype
                                            + "*"
                                            + str(len(tempval[0]))
                                            + ")(*"
                                            + str(tempval[0])
                                            + "),ctypes.POINTER(ctypes.c_"
                                            + temptype
                                            + ")))"
                                        )
                                    )
                                )
                            elif list(args.values())[i]["array"] == 1:
                                testsc.append(
                                    (eval("ctypes.c_" + temptype) * len(tempval))(
                                        *tempval
                                    )
                                )
                    else:
                        typelist.append(temptype)
                        testsp.append(data["success"]["num"])
                        testsc.append(data["success"]["num"])

                # Initialize test functions
                print("Testing ", name)
                print("----------------------------")
                funcPy = getattr(kernels, name)
                funcC = getattr(lib, name)
                funcC.restype = Error
                funcC.argtypes = getctypelist(typelist)

                # Success tests
                if name not in success_blacklist:
                    print()
                    print("Testing success")
                    print("Input Parameters - ")
                    for i in range(len(argnames)):
                        print(argnames[i], ": ", testsp[i])
                    _ = funcPy(*testsp)
                    ret_pass = funcC(*testsc)
                    print("Output Parameters - ")
                    for i in checkindex:
                        print("Param name: ", argnames[i])
                        print("Python \t C")
                        if isinstance(testsp[i], list):
                            for j in range(len(testsp[i])):
                                print(testsp[i][j], " \t", testsc[i][j])
                                assert math.isclose(
                                    testsp[i][j], testsc[i][j], rel_tol=0.0001
                                )
                        else:
                            print(testsp[i], " \t", testsc[i])
                            assert testsp[i] == testsc[i]
                    for i in pindex:
                        print("Param name: ", argnames[i])
                        print("Python \t C")
                        print(testsp[i][0], " \t", pval[i].value)
                        assert testsp[i][0] == pval[i].value
                    assert not ret_pass.str
                    print()

                # Only test failure if function raises ValueError
                if name in failfuncs and name not in failure_blacklist:
                    testsp = []
                    testsc = []
                    for i in range(len(args.values())):
                        if list(args.values())[i]["type"].endswith("_t"):
                            temptype = list(args.values())[i]["type"][:-2]
                        elif list(args.values())[i]["type"] == "int":
                            temptype = "int64"
                        else:
                            temptype = list(args.values())[i]["type"]
                        if list(args.values())[i]["check"] == "outparam":
                            temparr = [0] * (data["failure"]["num"] + 50)
                            testsp.append(temparr)
                            if (
                                "role" in list(args.values())[i]
                                and list(args.values())[i]["role"] == "pointer"
                            ):
                                testsc.append(ctypes.byref(pval[-1]))
                            else:
                                testsc.append(
                                    (eval("ctypes.c_" + temptype) * len(temparr))(
                                        *temparr
                                    )
                                )
                                checkindex.append(i)
                        elif "role" in list(args.values())[i]:
                            if "instance" in list(args.values())[i]:
                                tempval = data["failure"][
                                    list(args.values())[i]["role"][
                                        : list(args.values())[i]["role"].find("-")
                                    ]
                                ][list(args.values())[i]["role"]][
                                    int(list(args.values())[i]["instance"])
                                ][
                                    pytype(temptype)
                                ]
                            else:
                                tempval = data["failure"][
                                    list(args.values())[i]["role"][
                                        : list(args.values())[i]["role"].find("-")
                                    ]
                                ][list(args.values())[i]["role"]][0][pytype(temptype)]
                            testsp.append(tempval)
                            if not isinstance(tempval, list):
                                testsc.append(tempval)
                            else:
                                if list(args.values())[i]["array"] == 2:
                                    testsc.append(
                                        (
                                            eval(
                                                "ctypes.pointer(ctypes.cast((ctypes.c_"
                                                + temptype
                                                + "*"
                                                + str(len(tempval[0]))
                                                + ")(*"
                                                + str(tempval[0])
                                                + "),ctypes.POINTER(ctypes.c_"
                                                + temptype
                                                + ")))"
                                            )
                                        )
                                    )
                                elif list(args.values())[i]["array"] == 1:
                                    testsc.append(
                                        (eval("ctypes.c_" + temptype) * len(tempval))(
                                            *tempval
                                        )
                                    )
                        else:
                            testsp.append(data["failure"]["num"])
                            testsc.append(data["failure"]["num"])

                    # Failure tests
                    print("Testing failure")
                    print("Input Parameters - ")
                    for i in range(len(argnames)):
                        print(argnames[i], ": ", testsp[i])
                    try:
                        funcPy(*testsp)
                        raise AssertionError("funcPy should raise failure")
                    except ValueError:
                        pass
                    assert funcC(*testsc).str.contents
