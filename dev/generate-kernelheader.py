# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import os

import yaml

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def getctype(typename):
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


if __name__ == "__main__":
    with open(
        os.path.join(CURRENT_DIR, "..", "include", "awkward", "kernels.h"), "w"
    ) as header:
        header.write("// AUTO GENERATED: DO NOT EDIT BY HAND!\n")
        header.write(
            "// To regenerate file, execute - python dev/generate-kernelheader.py\n\n"
        )
        header.write(
            '#ifndef AWKWARD_KERNELS_H_\n#define AWKWARD_KERNELS_H_\n\n#include "awkward/common.h"\n\nextern "C" {\n'
        )
        with open(
            os.path.join(CURRENT_DIR, "..", "kernel-specification.yml")
        ) as specfile:
            indspec = yaml.safe_load(specfile)["kernels"]
            for spec in indspec:
                for childfunc in spec["specializations"]:
                    header.write(" " * 2 + "EXPORT_SYMBOL ERROR\n")
                    header.write(" " * 2 + childfunc["name"] + "(\n")
                    for i, arg in enumerate(childfunc["args"]):
                        header.write(
                            " " * 4 + getctype(arg["type"]) + " " + arg["name"]
                        )
                        if i == (len(childfunc["args"]) - 1):
                            header.write(");\n")
                        else:
                            header.write(",\n")
                header.write("\n")
        header.write("}\n#endif\n")
