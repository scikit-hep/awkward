# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import argparse
import pkg_resources

if __name__ == "__main__":
    awkward_path = pkg_resources.resource_filename("awkward1", "")
    libawkward_cpu_kernels_name = "libawkward-cpu-kernels.so"
    libawkward_name = "libawkward.so"

    argparser = argparse.ArgumentParser(description="Print out compilation arguments to use Awkward Array as a C++ dependency")
    argparser.add_argument("--cflags", action="store_true", help="output compiler flags and Awkward include path")
    argparser.add_argument("--libs", action="store_true", help="output Awkward libraries with path")
    argparser.add_argument("--libs-only-L", action="store_true", help="output Awkward library path")
    argparser.add_argument("--libs-only-l", action="store_true", help="output Awkward libraries without path")
    argparser.add_argument("--cflags-only-I", action="store_true", help="output Awkward include path")

    # only used in validating the arguments
    args = argparser.parse_args()

    output = []

    # loop over original sys.argv to get optional arguments in order
    for arg in sys.argv:
        if arg == "--cflags":
            output.append("-std=c++11 -I{0}".format(
                pkg_resources.resource_filename("awkward1", "include")
            ))

        if arg == "--libs":
            output.append("-L{0} -l{1} -l{2}".format(
                pkg_resources.resource_filename("awkward1", ""),
                libawkward_name,
                libawkward_cpu_kernels_name,
            ))

        if arg == "--libs-only-L":
            output.append("-L{0}".format(
                pkg_resources.resource_filename("awkward1", ""),
            ))

        if arg == "--libs-only-l":
            output.append("-l{0} -l{1}".format(
                libawkward_name,
                libawkward_cpu_kernels_name,
            ))

        if arg == "--cflags-only-I":
            output.append("-I{0}".format(
                pkg_resources.resource_filename("awkward1", "include")
            ))

    print(" ".join(output))
