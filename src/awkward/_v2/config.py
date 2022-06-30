# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import sys
import os
import argparse
import pkg_resources

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Print out compilation arguments to use Awkward Array as a C++ dependency"
    )
    argparser.add_argument(
        "--cflags",
        action="store_true",
        help="output compiler flags and Awkward include path",
    )
    argparser.add_argument(
        "--cflags-only-I", action="store_true", help="output Awkward include path"
    )
    argparser.add_argument(
        "--incdir", action="store_true", help="output Awkward include directory name"
    )

    # only used in validating the arguments
    args = argparser.parse_args()

    output = []
    incdir = pkg_resources.resource_filename(
        "awkward", os.path.join("src", "_v2", "cpp-headers")
    )

    # loop over original sys.argv to get optional arguments in order
    for arg in sys.argv:
        if arg == "--cflags":
            output.append(f"-std=c++17 -I{incdir}")

        if arg == "--cflags-only-I":
            output.append(f"-I{incdir}")

        if arg == "--incdir":
            output.append(incdir)

    print(" ".join(output))  # noqa: T201
