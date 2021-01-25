import argparse
import glob
import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def clean_tests():
    if (os.path.exists(os.path.join(CURRENT_DIR, "..", "tests-spec", "kernels.py"))):
        os.remove(os.path.join(CURRENT_DIR, "..", "tests-spec", "kernels.py"))
    cpu_kernel_tests = glob.glob(
        os.path.join(CURRENT_DIR, "..", "tests-cpu-kernels", "test") + "*"
    )
    for testfile in cpu_kernel_tests:
        os.remove(testfile)
    cuda_kernel_tests = glob.glob(
        os.path.join(CURRENT_DIR, "..", "tests-cuda-kernels", "test") + "*"
    )
    for testfile in cuda_kernel_tests:
        os.remove(testfile)
    kernel_spec_tests = glob.glob(
        os.path.join(CURRENT_DIR, "..", "tests-spec", "test") + "*"
    )
    for testfile in kernel_spec_tests:
        os.remove(testfile)


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
