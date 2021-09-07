# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import argparse
import glob
import os
import shutil

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def clean_tests():
    tests_spec = os.path.join(CURRENT_DIR, "..", "tests-spec")
    if os.path.exists(tests_spec):
        shutil.rmtree(tests_spec)

    tests_spec_explicit = os.path.join(CURRENT_DIR, "..", "tests-spec-explicit")
    if os.path.exists(tests_spec_explicit):
        shutil.rmtree(tests_spec_explicit)

    tests_cpu_kernels = os.path.join(CURRENT_DIR, "..", "tests-cpu-kernels")
    if os.path.exists(tests_cpu_kernels):
        shutil.rmtree(tests_cpu_kernels)

    tests_cuda_kernels = os.path.join(CURRENT_DIR, "..", "tests-cuda-kernels")
    if os.path.exists(tests_cuda_kernels):
        shutil.rmtree(tests_cuda_kernels)


def clean_cuda_kernels():
    cuda_kernels = glob.glob(
        os.path.join(CURRENT_DIR, "..", "src", "cuda-kernels", "awkward_") + "*.cu"
    )
    for kernel in cuda_kernels:
        os.remove(kernel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup generated files")
    parser.add_argument("--tests", default=True)
    parser.add_argument("--cuda-kernels", default=True)
    args = parser.parse_args()
    if args.tests:
        clean_tests()
    if args.cuda_kernels:
        clean_cuda_kernels()
