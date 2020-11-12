# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import argparse
import copy
import os
import sys
from collections import OrderedDict

import yaml

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def parse_spec():
    specdict = OrderedDict()
    with open(os.path.join(CURRENT_DIR, "..", "kernel-specification.yml")) as specfile:
        indspec = yaml.safe_load(specfile)["kernels"]
        for spec in indspec:
            specdict[spec["name"]] = []
            for childfunc in spec["specializations"]:
                specdict[spec["name"]].append(childfunc["name"])
    return specdict


def sort_specializations(keystring):
    ordering = [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "u8",
        "uint8",
        "u16",
        "uint16",
        "u32",
        "uint32",
        "u64",
        "uint64",
        "float16",
        "float32",
        "float64",
        "float128",
        "complex64",
        "complex128",
        "complex256",
    ]
    elemsfound = []
    keystring = keystring.lower()
    while any(element in keystring for element in ordering):
        for i, element in enumerate(ordering):
            if element in keystring and not (
                element.startswith("int")
                and keystring[keystring.find(element) - 1] == "u"
            ):
                elemsfound.append((keystring.find(element), i))
                keystring = keystring.replace(element, "", 1)
    elemsfound.sort()
    if len(elemsfound) == 0:
        return keystring
    elif len(elemsfound) == 1:
        return (elemsfound[0][1], 0)
    else:
        return (elemsfound[0][1], elemsfound[1][1])


def check_specorder(kerneldict):
    print("Checking kernel specification order")
    kernelnames = []
    with open(os.path.join(CURRENT_DIR, "..", "kernel-specification.yml")) as specfile:
        indspec = yaml.safe_load(specfile)["kernels"]
        for spec in indspec:
            kernelnames.append(spec["name"])
    count = 0
    for dictkernel in kerneldict.keys():
        if dictkernel != kernelnames[count]:
            print("Order in specification: ", kerneldict.keys())
            print("Sorted order: ", kernelnames)
            raise Exception("Kernels not sorted in specification")
        count += 1
    for kernel, specializations in kerneldict.items():
        display = []
        try:
            count = 0
            display = []
            flag = False
            if sys.version_info.major == 2:
                for dummy in specializations:
                    if type(sort_specializations(dummy)) == str:
                        raise TypeError
            for specialization in sorted(
                copy.copy(specializations), key=sort_specializations
            ):
                if specialization != specializations[count]:
                    flag = True
                display.append(specialization)
                count += 1
        except TypeError:
            count = 0
            display = []
            flag = False
            for specialization in sorted(copy.copy(specializations)):
                if specialization != specializations[count]:
                    flag = True
                display.append(specialization)
                count += 1
        if flag:
            print("For kernel: " + kernel)
            print("Order in specification = ", specializations)
            print("Sorted order = ", display)
            raise Exception("Kernel specializations not sorted in specification")
    print("Kernel specification file is properly sorted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Awkward Kernel Diagnostics Tool")
    parser.add_argument(
        "--check-sorted",
        action="store_true",
        help="Check if kernel specification file is sorted",
    )
    args = parser.parse_args()
    kernels = parse_spec()
    if args.check_sorted:
        check_specorder(kernels)
