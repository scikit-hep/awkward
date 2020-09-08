# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import argparse
import pkg_resources

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Print out compilation arguments to use Awkward Array as a C++ dependency")
    argparser.add_argument("--cflags", action="store_true", help="output compiler flags and Awkward include path")
    argparser.add_argument("--libs", action="store_true", help="output Awkward libraries with path")
    argparser.add_argument("--libs-only-L", action="store_true", help="output Awkward library path")
    argparser.add_argument("--libs-only-l", action="store_true", help="output Awkward libraries without path")
    argparser.add_argument("--static-libs", action="store_true", help="output Awkward static libraries with path")
    argparser.add_argument("--static-libs-only-L", action="store_true", help="output Awkward static library path")
    argparser.add_argument("--static-libs-only-l", action="store_true", help="output Awkward static libraries without path")
    argparser.add_argument("--cflags-only-I", action="store_true", help="output Awkward include path")
    argparser.add_argument("--incdir", action="store_true", help="output Awkward include directory name")
    argparser.add_argument("--libdir", action="store_true", help="output Awkward library directory name")

    # only used in validating the arguments
    args = argparser.parse_args()

    output = []
    incdir = pkg_resources.resource_filename("awkward1", "include")
    libdir = pkg_resources.resource_filename("awkward1", "")
    cpu_kernels = "awkward-cpu-kernels"
    libawkward = "awkward"

    # loop over original sys.argv to get optional arguments in order
    for arg in sys.argv:
        if arg == "--cflags":
            output.append("-std=c++11 -I{0}".format(incdir))

        if arg == "--libs":
            output.append("-L{0} -l{1} -l{2} -ldl".format(libdir, libawkward, cpu_kernels))

        if arg == "--libs-only-L":
            output.append("-L{0}".format(libdir))

        if arg == "--libs-only-l":
            output.append("-l{0} -l{1} -ldl".format(libawkward, cpu_kernels))

        if arg == "--static-libs":
            output.append("-L{0} -l{1}-static -l{2}-static -ldl".format(libdir, libawkward, cpu_kernels))

        if arg == "--static-libs-only-L":
            output.append("-L{0}".format(libdir))

        if arg == "--static-libs-only-l":
            output.append("-l{0}-static -l{1}-static -ldl".format(libawkward, cpu_kernels))

        if arg == "--cflags-only-I":
            output.append("-I{0}".format(incdir))

        if arg == "--incdir":
            output.append(incdir)

        if arg == "--libdir":
            output.append(libdir)

    print(" ".join(output))
