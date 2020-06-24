# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import argparse
import os

import kernelparser


def genpykernels():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("filenames", nargs="+")
    args = arg_parser.parse_args()
    filenames = args.filenames
    fillerpy = """import copy

kMaxInt64  = 9223372036854775806
kSliceNone = kMaxInt64 + 1

outparam = None
inparam = None

"""
    _, gencode = kernelparser.parser(filenames)
    gencode = fillerpy + gencode
    current_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(current_dir, "kernels.py"), "w") as f:
        print("Writing kernels.py")
        f.write(gencode)


if __name__ == "__main__":
    genpykernels()
