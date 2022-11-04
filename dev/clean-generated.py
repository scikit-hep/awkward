# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import argparse
import glob
import os
import shutil

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def clean_tests():
    tests_spec = os.path.join(CURRENT_DIR, "..", "awkward-cpp", "tests-spec")
    if os.path.exists(tests_spec):
        shutil.rmtree(tests_spec)

    tests_spec_explicit = os.path.join(
        CURRENT_DIR, "..", "awkward-cpp", "tests-spec-explicit"
    )
    if os.path.exists(tests_spec_explicit):
        shutil.rmtree(tests_spec_explicit)

    tests_cpu_kernels = os.path.join(
        CURRENT_DIR, "..", "awkward-cpp", "tests-cpu-kernels"
    )
    if os.path.exists(tests_cpu_kernels):
        shutil.rmtree(tests_cpu_kernels)

    tests_cuda_kernels = os.path.join(CURRENT_DIR, "..", "tests-cuda-kernels")
    if os.path.exists(tests_cuda_kernels):
        shutil.rmtree(tests_cuda_kernels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup generated files")
    parser.add_argument("--tests", default=True)
    args = parser.parse_args()
    if args.tests:
        clean_tests()
