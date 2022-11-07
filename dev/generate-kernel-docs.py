# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import os

import yaml

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def indent_code(code, indent):
    finalcode = ""
    for line in code.splitlines():
        finalcode += " " * indent + line + "\n"
    return finalcode


def genkerneldocs():
    prefix = """Kernel interface and specification
----------------------------------

All array manipulation takes place in the lowest layer of the Awkward Array project, the "kernels." The primary implementation of these kernels are in ``libawkward-cpu-kernels.so`` (or similar names on MacOS and Windows), which has a pure C interface.

A second implementation, ``libawkward-cuda-kernels.so``, is provided as a separate package, ``awkward-cuda-kernels``, which handles arrays that reside on GPUs if CUDA is available. It satisfies the same C interface and implements the same behaviors.

.. figure:: ../../image/awkward-1-0-layers.svg
   :align: center


The functions are implemented in C with templates for integer specializations (cpu-kernels) and as CUDA (cuda-kernels), but the function signatures and normative definitions are expressed below using a subset of the Python language. These normative definitions are used as a stable and easy-to-read standard that both implementations must reproduce in tests, regardless of how they are optimized.\n
"""
    with open(
        os.path.join(CURRENT_DIR, "..", "awkward-cpp", "kernel-specification.yml")
    ) as specfile:
        with open(
            os.path.join(
                CURRENT_DIR,
                "..",
                "docs",
                "reference",
                "generated",
                "kernels.rst",
            ),
            "w",
        ) as outfile:
            outfile.write(prefix)
            indspec = yaml.safe_load(specfile)["kernels"]
            for spec in indspec:
                outfile.write(spec["name"] + "\n")
                print("Generating doc for " + spec["name"])
                outfile.write(
                    "========================================================================\n"
                )
                for childfunc in spec["specializations"]:
                    outfile.write(".. py:function:: " + childfunc["name"])
                    outfile.write("(")
                    for i in range(len(childfunc["args"])):
                        if i != 0:
                            outfile.write(
                                ", "
                                + childfunc["args"][i]["name"]
                                + ": "
                                + childfunc["args"][i]["type"]
                            )
                        else:
                            outfile.write(
                                childfunc["args"][i]["name"]
                                + ": "
                                + childfunc["args"][i]["type"]
                            )
                    outfile.write(")\n")
                outfile.write(".. code-block:: python\n\n")
                # Remove conditional at the end of dev
                if "def" in spec["definition"]:
                    outfile.write(
                        indent_code(
                            spec["definition"],
                            4,
                        )
                        + "\n\n"
                    )


if __name__ == "__main__":
    genkerneldocs()
