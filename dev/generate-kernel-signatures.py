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
    "awkward_new_Identities",
    "awkward_Identities32_to_Identities64",
    "awkward_ListOffsetArray_flatten_offsets",
    "awkward_IndexedArray_overlay_mask",
    "awkward_IndexedArray_mask",
    "awkward_ByteMaskedArray_mask",
    "awkward_zero_mask",
    "awkward_RegularArray_compact_offsets",
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
    "awkward_RegularArray_broadcast_tooffsets",
    "awkward_NumpyArray_fill_tobool",
    "awkward_NumpyArray_reduce_adjust_starts_64",
    "awkward_NumpyArray_reduce_adjust_starts_shifts_64",
    "awkward_regularize_arrayslice",
    "awkward_RegularArray_getitem_next_at",
    "awkward_ListOffsetArray_compact_offsets",
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
    "awkward_reduce_sum_int32_bool_64" "awkward_reduce_sum_int64_bool_64",
    "awkward_reduce_sum_bool",
    "awkward_reduce_prod_bool",
    "awkward_reduce_countnonzero",
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

# pylint: skip-file

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

from awkward._v2._connect.cuda import fetch_specialization
from awkward._v2._connect.cuda import import_cupy

cupy = import_cupy("Awkward Arrays with CUDA")
"""
        )

        file.write(
            """
def by_signature(cuda_kernel_templates):
    out = {}
"""
        )
        with open(
            os.path.join(
                os.path.dirname(CURRENT_DIR),
                "src",
                "awkward",
                "_v2",
                "_connect",
                "cuda",
                "cuda_kernels",
                "cuda_common.cu",
            ),
        ) as cu_file:
            code = cu_file.read()
            python_code = code[
                code.find("// BEGIN PYTHON") : code.find("// END PYTHON")
            ]
            python_code = python_code.replace("// BEGIN PYTHON", "")
            python_code = python_code.replace("// ", "    ")
            file.write(python_code)

        for spec in specification["kernels"]:
            for childfunc in spec["specializations"]:
                special = [repr(spec["name"])]
                [type_to_pytype(x["type"], special) for x in childfunc["args"]]
                dirlist = [repr(x["dir"]) for x in childfunc["args"]]
                if spec["name"] in cuda_kernels_impl:
                    with open(
                        os.path.join(
                            os.path.dirname(CURRENT_DIR),
                            "src",
                            "awkward",
                            "_v2",
                            "_connect",
                            "cuda",
                            "cuda_kernels",
                            spec["name"] + ".cu",
                        ),
                    ) as cu_file:
                        code = cu_file.read()

                        if "// BEGIN PYTHON" not in code:
                            file.write(
                                """
    def f(grid, block, args):
        cuda_kernel_templates.get_function(fetch_specialization([{}]))(grid, block, args)
    f.dir = [{}]
    out[{}] = f
""".format(
                                    ", ".join(special),
                                    ", ".join(dirlist),
                                    ", ".join(special),
                                )
                            )
                        else:

                            python_code = code[
                                code.find("// BEGIN PYTHON") : code.find(
                                    "// END PYTHON"
                                )
                            ]
                            python_code = python_code.replace("// BEGIN PYTHON", "")
                            python_code = python_code.replace("// ", "    ")

                            if "{dtype_specializations}" in python_code:
                                python_code = python_code.replace(
                                    "{dtype_specializations}", ", ".join(special[1:])
                                )

                            file.write(python_code)
                            file.write(
                                """    f.dir = [{}]
    out[{}] = f
""".format(
                                    ", ".join(dirlist), ", ".join(special)
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


if __name__ == "__main__":
    with open(os.path.join(CURRENT_DIR, "..", "kernel-specification.yml")) as specfile:
        specification = yaml.safe_load(specfile)
        include_kernels_h(specification)
        kernel_signatures_py(specification)
        kernel_signatures_cuda_py(specification)
