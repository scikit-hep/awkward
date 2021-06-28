# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import os
import datetime

import yaml

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def type_to_ctype(typename):
    flag = False
    if "Const[" in typename:
        flag = True
        typename = typename[len("Const[") : -1]
    arraycount = 0
    while "List[" in typename:
        arraycount += 1
        typename = typename[len("List[") : -1]
    typename = typename + "*" * arraycount
    if flag:
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
                datetime.datetime.now().isoformat().replace("T", " AT ")[:22]
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


if __name__ == "__main__":
    with open(os.path.join(CURRENT_DIR, "..", "kernel-specification.yml")) as specfile:
        specification = yaml.safe_load(specfile)
        include_kernels_h(specification)
