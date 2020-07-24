# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import os

import yaml

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def genpykernels():
    prefix = """
import copy

kMaxInt64  = 9223372036854775806
kSliceNone = kMaxInt64 + 1

"""
    with open(os.path.join(CURRENT_DIR, "spec.yaml")) as infile:
        spec = yaml.safe_load(infile)["Kernels"]
        with open(os.path.join(CURRENT_DIR, "kernels.py"), "w") as outfile:
            outfile.write(prefix)
            for func in spec:
                outfile.write(func["specification"])
                if "specializations" in func.keys():
                    for childfunc in func["specializations"]:
                        outfile.write(childfunc["name"] + " = " + func["name"] + "\n")
                    outfile.write("\n\n")
