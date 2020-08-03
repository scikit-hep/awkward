# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import os

import yaml

from parser_utils import indent_code

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def genkerneldocs():
    prefix = """Kernel interface and specification
----------------------------------

All array manipulation takes place in the lowest layer of the Awkward Array project, the "kernels." The primary implementation of these kernels are in ``libawkward-cpu-kernels.so`` (or similar names on MacOS and Windows), which has a pure C interface.

A second implementation, ``libawkward-cuda-kernels.so``, is provided as a separate package, ``awkward1-cuda``, which handles arrays that reside on GPUs if CUDA is available. It satisfies the same C interface and implements the same behaviors.

.. raw:: html

    <img src="../_static/awkward-1-0-layers.svg" style="max-width: 500px; margin-left: auto; margin-right: auto;">

The interface, as well as specifications for each function's behavior through a normative Python implementation, are presented below.

"""
    with open(
        os.path.join(CURRENT_DIR, "..", "kernel-specification", "kernelnames.yml")
    ) as infile:
        mainspec = yaml.safe_load(infile)["kernels"]
        with open(
            os.path.join(CURRENT_DIR, "..", "docs-sphinx", "_auto", "kernels.rst",), "w"
        ) as outfile:
            outfile.write(prefix)
            for filedir in mainspec.values():
                for relpath in filedir.values():
                    with open(
                        os.path.join(CURRENT_DIR, "..", "kernel-specification", relpath)
                    ) as specfile:
                        indspec = yaml.safe_load(specfile)[0]
                        outfile.write(indspec["name"] + "\n")
                        print("Generating doc for " + indspec["name"])
                        outfile.write(
                            "=================================================================\n"
                        )
                        if "specializations" in indspec.keys():
                            for childfunc in indspec["specializations"]:
                                outfile.write(".. py:function:: " + childfunc["name"])
                                outfile.write("(")
                                for i in range(len(childfunc["args"])):
                                    if i != 0:
                                        outfile.write(
                                            ", "
                                            + list(childfunc["args"][i].keys())[0]
                                            + ": "
                                            + list(childfunc["args"][i].values())[0]
                                        )
                                    else:
                                        outfile.write(
                                            list(childfunc["args"][i].keys())[0]
                                            + ": "
                                            + list(childfunc["args"][i].values())[0]
                                        )
                                outfile.write(")\n")
                        outfile.write(".. code-block:: python\n\n")
                        # Remove conditional at the end of dev
                        if "def" in indspec["definition"]:
                            outfile.write(
                                indent_code(indspec["definition"], 4,) + "\n\n"
                            )

        if os.path.isfile(
            os.path.join(CURRENT_DIR, "..", "docs-sphinx", "_auto", "toctree.txt",)
        ):
            with open(
                os.path.join(CURRENT_DIR, "..", "docs-sphinx", "_auto", "toctree.txt",),
                "r+",
            ) as f:
                if "_auto/kernels.rst" not in f.read():
                    print("Updating toctree.txt")
                    f.write(" " * 4 + "_auto/kernels.rst")


if __name__ == "__main__":
    genkerneldocs()
