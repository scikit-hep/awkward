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

All array manipulation that is not performed by NumPy/CuPy/etc. is executed in Awkward Array's low-level "kernels." These functions communicate entirely by side-effects, manipulating already-allocated arrays, and therefore can be implemented with a pure C interface. They're called "kernels" because their signatures are similar to GPU kernels, to make them easier to port to GPUs and similar devices.

All of the kernel functions that Awkward Array uses are documented below. These are internal details of Awkward Array, subject to change at any time, not a public API. The reason that we are documenting the API is to ensure compatibility between the primary CPU-bound kernels and their equivalents, ported to GPUs (and similar devices). The definitions below are expressed in Python code, but the implementations used by Awkward Array are compiled.

"""
    generated_dir = os.path.join(CURRENT_DIR, "..", "docs", "reference", "generated")
    os.makedirs(generated_dir, exist_ok=True)

    with open(os.path.join(CURRENT_DIR, "..", "kernel-specification.yml")) as specfile:
        with open(
            os.path.join(
                generated_dir,
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
