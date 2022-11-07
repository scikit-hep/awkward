# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def clean_kernel_docs():
    kernel_docs = os.path.join(
        CURRENT_DIR, "..", "docs-sphinx", "reference", "generated", "kernels.rst"
    )
    if os.path.exists(kernel_docs):
        os.unlink(kernel_docs)


if __name__ == "__main__":
    clean_kernel_docs()
