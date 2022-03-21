# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import os
import datetime
import time

import yaml

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


cuda_kernels_impl = [
    "awkward_ListArray_num",
    "awkward_RegularArray_num",
    "awkward_ListArray_validity",
    "awkward_BitMaskedArray_to_ByteMaskedArray",
    "awkward_ListArray_compact_offsets",
    "awkward_ListOffsetArray_compact_offsets",
]


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
    f = lib.{}
    f.argtypes = [{}]
    f.restype = ERROR
    f.dir = [{}]
    out[{}] = f
""".format(
                        str(childfunc["name"]),
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


def kernel_signatures_cuda_py(specification):
    print("Generating src/awkward/_connect/cuda/_kernel_signatures.py...")

    with open(
        os.path.join(
            os.path.dirname(CURRENT_DIR),
            "src",
            "awkward",
            "_v2",
            "_connect",
            "cuda",
            "_kernel_signatures.py",
        ),
        "w",
    ) as file:
        file.write(
            f"""# AUTO GENERATED ON {reproducible_datetime()}
# DO NOT EDIT BY HAND!
#
# To regenerate file, run
#
#     python dev/generate-kernel-signatures.py
#
# (It is usually run as part of pip install . or localbuild.py.)

# fmt: off

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

kernel_specializations = {{
"""
        )
        kernel_specializations = generate_kernel_specializations(specification)
        file.write("    " + kernel_specializations[1:-1] + "\n}\n")

        file.write(
            """
def by_signature(cuda_kernel_templates):
    out = {}
"""
        )

        for spec in specification["kernels"]:
            for childfunc in spec["specializations"]:
                special = [repr(spec["name"])]
                [type_to_pytype(x["type"], special) for x in childfunc["args"]]
                dirlist = [repr(x["dir"]) for x in childfunc["args"]]
                if spec["name"] in cuda_kernels_impl:
                    file.write(
                        """
    f = lambda: cuda_kernel_templates.get_function(kernel_specializations[{}])
    f.dir = [{}]
    out[{}] = f
""".format(
                            repr(childfunc["name"]),
                            ", ".join(dirlist),
                            ", ".join(special),
                        )
                    )
                else:
                    file.write(
                        """
    out[{}] = None
""".format(
                            ", ".join(special),
                        )
                    )

        file.write(
            """
    return out
"""
        )

    print("Done with  src/awkward/_kernel_signatures_cuda.py...")


def generate_kernel_specializations(specification):
    import re

    kernel_specializations = {}
    failed_cases = []
    for spec in specification["kernels"]:
        if spec["name"] in cuda_kernels_impl:
            filename = spec["name"] + ".cpp"
            try:
                with open(
                    os.path.join(
                        os.path.dirname(CURRENT_DIR), "src", "cpu-kernels", filename
                    )
                ) as kernel:
                    code = kernel.read()
                    for childfunc in spec["specializations"]:
                        kernel_specialized_name = childfunc["name"]
                        result_errstring = [
                            _.start()
                            for _ in re.finditer(kernel_specialized_name, code)
                        ]
                        c_specialization = code[result_errstring[0] :]
                        if c_specialization.find(spec["name"] + "<") == -1:
                            kernel_specializations[kernel_specialized_name] = spec[
                                "name"
                            ]
                            continue
                        c_specialization = c_specialization[
                            c_specialization.find(
                                spec["name"] + "<"
                            ) : c_specialization.find(">")
                            + 1
                        ]
                        kernel_specializations[
                            kernel_specialized_name
                        ] = c_specialization

            except FileNotFoundError:
                failed_cases.append(spec["name"])
                pass

    print("Couldn't Generate Specializations for: ", failed_cases)

    return kernel_specializations.__repr__().replace(", '", ",\n    '")


if __name__ == "__main__":
    with open(os.path.join(CURRENT_DIR, "..", "kernel-specification.yml")) as specfile:
        specification = yaml.safe_load(specfile)
        include_kernels_h(specification)
        kernel_signatures_py(specification)
        kernel_signatures_cuda_py(specification)
