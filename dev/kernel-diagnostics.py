# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

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
    print("Checking kernel specification order...")
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


def check_spec_implementation():
    count = 0
    with open(os.path.join(CURRENT_DIR, "..", "kernel-specification.yml")) as specfile:
        indspec = yaml.safe_load(specfile)["kernels"]
        for spec in indspec:
            if "def awkward" not in spec["definition"]:
                if count == 0:
                    print("\nKernels not implemented in specification file - ")
                print(spec["name"])
                count += 1


def check_cpu_implementation(kerneldict):
    count = 0
    for kernelname, specializations in kerneldict.items():
        if os.path.isfile(
            os.path.join(CURRENT_DIR, "..", "src", "cpu-kernels", kernelname + ".cpp")
        ):
            with open(
                os.path.join(
                    CURRENT_DIR, "..", "src", "cpu-kernels", kernelname + ".cpp"
                )
            ) as kernelfile:
                contents = kernelfile.read()
                for childname in specializations:
                    if childname not in contents:
                        if count == 0:
                            print("\nKernels not implemented as a CPU kernel - ")
                        print(kernelname + ": " + childname)
                        count += 1
        else:
            if count == 0:
                print("\nKernels not implemented as a CPU kernel - ")
            print(kernelname)
            count += 1


def check_cuda_implementation(kerneldict):
    count = 0
    for kernelname in kerneldict.keys():
        if not (
            os.path.isfile(
                os.path.join(
                    CURRENT_DIR, "..", "src", "cuda-kernels", kernelname + ".cu"
                )
            )
            or os.path.isfile(
                os.path.join(
                    CURRENT_DIR,
                    "..",
                    "src",
                    "cuda-kernels",
                    "manual_" + kernelname + ".cu",
                )
            )
        ):
            if count == 0:
                print("\nKernels not implemented as a CUDA kernel - ")
            print(kernelname)
            count += 1


def check_implementations(kerneldict):
    print("Checking if kernels are implemented everywhere...")
    check_spec_implementation()
    check_cpu_implementation(kerneldict)
    check_cuda_implementation(kerneldict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Awkward Kernel Diagnostics Tool")
    parser.add_argument(
        "--check-spec-sorted",
        action="store_true",
        help="Check if kernel specification file is sorted",
    )
    parser.add_argument(
        "--check-implemented",
        action="store_true",
        help="Check if kernel is present in specification file, as a CPU kernel and as a CUDA kernel",
    )
    args = parser.parse_args()
    kernels = parse_spec()
    count = 0
    if args.check_spec_sorted:
        check_specorder(kernels)
        count += 1
    if args.check_implemented:
        if count != 0:
            print("\n")
        check_implementations(kernels)
