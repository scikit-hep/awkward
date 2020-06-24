# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import argparse
import os
import sys

import kernelparser

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("filenames", nargs="+")
    args = arg_parser.parse_args()
    filenames = args.filenames
    docdict, _ = kernelparser.parser(filenames)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if os.path.isdir(os.path.join(current_dir, "..", "docs-sphinx", "_auto")):
        with open(
            os.path.join(current_dir, "..", "docs-sphinx", "_auto", "kernels.rst",),
            "w",
        ) as f:
            print("Writing kernels.rst")
            f.write(
                """Kernel interface and specification
----------------------------------

All array manipulation takes place in the lowest layer of the Awkward Array project, the "kernels." The primary implementation of these kernels are in ``libawkward-cpu-kernels.so`` (or similar names on MacOS and Windows), which has a pure C interface.

A second implementation, ``libawkward-cuda-kernels.so``, is provided as a separate package, ``awkward1-cuda``, which handles arrays that reside on GPUs if CUDA is available. It satisfies the same C interface and implements the same behaviors.

.. raw:: html

    <img src="../_static/awkward-1-0-layers.svg" style="max-width: 500px; margin-left: auto; margin-right: auto;">

The interface, as well as specifications for each function's behavior through a normative Python implementation, are presented below.

"""
            )
            for name in sorted(docdict.keys()):
                f.write(docdict[name])
        if os.path.isfile(
            os.path.join(current_dir, "..", "docs-sphinx", "_auto", "toctree.txt",)
        ):
            with open(
                os.path.join(current_dir, "..", "docs-sphinx", "_auto", "toctree.txt",),
                "r+",
            ) as f:
                if "_auto/kernels.rst" not in f.read():
                    print("Updating toctree.txt")
                    f.write(" " * 4 + "_auto/kernels.rst")
