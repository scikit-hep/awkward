# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import os
import datetime
import time

import yaml

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def reproducible_datetime():

    build_date = datetime.datetime.utcfromtimestamp(
        int(os.environ.get("SOURCE_DATE_EPOCH", time.time()))
    )
    return build_date.isoformat().replace("T", " AT ")[:22]


def type_to_ctype(typename):
    is_const = False
    if "Const[" in typename:
        is_const = True
        typename = typename[len("Const[") : -1]
    count = 0
    while "List[" in typename:
        count += 1
        typename = typename[len("List[") : -1]
    typename = typename + "*" * count
    if is_const:
        typename = "const " + typename
    return typename


def include_kernels_h(specification):
    print("Generating include/awkward/kernels.h...")

    with open(
        os.path.join(CURRENT_DIR, "..", "include", "awkward", "kernels.h"), "w"
    ) as header:
        header.write(
            """// AUTO GENERATED ON {0}
// DO NOT EDIT BY HAND!
//
// To regenerate file, run
//
//     python dev/generate-kernel-signatures.py
//
// (It is usually run as part of pip install . or localbuild.py.)

#ifndef AWKWARD_KERNELS_H_
#define AWKWARD_KERNELS_H_

#include "awkward/common.h"

extern "C" {{

""".format(
                reproducible_datetime()
            )
        )
        for spec in specification["kernels"]:
            for childfunc in spec["specializations"]:
                header.write(" " * 2 + "EXPORT_SYMBOL ERROR\n")
                header.write(" " * 2 + childfunc["name"] + "(\n")
                for i, arg in enumerate(childfunc["args"]):
                    header.write(
                        " " * 4 + type_to_ctype(arg["type"]) + " " + arg["name"]
                    )
                    if i == (len(childfunc["args"]) - 1):
                        header.write(");\n")
                    else:
                        header.write(",\n")
            header.write("\n")
        header.write(
            """}

#endif // AWKWARD_KERNELS_H_
"""
        )

    print("Done with  include/awkward/kernels.h.")


type_to_dtype = {
    "bool": "bool_",
    "int8": "int8",
    "uint8": "uint8",
    "int16": "int16",
    "uint16": "uint16",
    "int32": "int32",
    "uint32": "uint32",
    "int64": "int64",
    "uint64": "uint64",
    "float": "float32",
    "double": "float64",
}


def type_to_pytype(typename, special):
    if "Const[" in typename:
        typename = typename[len("Const[") : -1]
    count = 0
    while "List[" in typename:
        count += 1
        typename = typename[len("List[") : -1]
    if typename.endswith("_t"):
        typename = typename[:-2]
    if count != 0:
        special.append(type_to_dtype[typename])
    return ("POINTER(" * count) + ("c_" + typename) + (")" * count)


def kernel_signatures_py(specification):
    print("Generating src/awkward/_kernel_signatures.py...")

    with open(
        os.path.join(CURRENT_DIR, "..", "src", "awkward", "_kernel_signatures.py"),
        "w",
    ) as file:
        file.write(
            """# AUTO GENERATED ON {0}
# DO NOT EDIT BY HAND!
#
# To regenerate file, run
#
#     python dev/generate-kernel-signatures.py
#
# (It is usually run as part of pip install . or localbuild.py.)

# fmt: off

from ctypes import (
    POINTER,
    Structure,
    c_bool,
    c_int8,
    c_uint8,
    c_int16,
    c_uint16,
    c_int32,
    c_uint32,
    c_int64,
    c_uint64,
    c_float,
    c_double,
    c_char_p,
)

import numpy as np

from numpy import (
    bool_,
    int8,
    uint8,
    int16,
    uint16,
    int32,
    uint32,
    int64,
    uint64,
    float32,
    float64,
)

class ERROR(Structure):
    _fields_ = [
        ("str", c_char_p),
        ("filename", c_char_p),
        ("id", c_int64),
        ("attempt", c_int64),
        ("pass_through", c_bool),
    ]


def by_signature(lib):
    out = {{}}
""".format(
                reproducible_datetime()
            )
        )

        for spec in specification["kernels"]:
            for childfunc in spec["specializations"]:
                special = [repr(spec["name"])]
                arglist = [
                    type_to_pytype(x["type"], special) for x in childfunc["args"]
                ]
                dirlist = [repr(x["dir"]) for x in childfunc["args"]]
                file.write(
                    """
    f = lib.{0}
    f.argtypes = [{1}]
    f.restype = ERROR
    f.dir = [{2}]
    out[{3}] = f
""".format(
                        childfunc["name"],
                        ", ".join(arglist),
                        ", ".join(dirlist),
                        ", ".join(special),
                    )
                )

        file.write(
            """
    return out
"""
        )

    print("Done with  src/awkward/_kernel_signatures.py...")


if __name__ == "__main__":
    with open(os.path.join(CURRENT_DIR, "..", "kernel-specification.yml")) as specfile:
        specification = yaml.safe_load(specfile)
        include_kernels_h(specification)
        kernel_signatures_py(specification)
