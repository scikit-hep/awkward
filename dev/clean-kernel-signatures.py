# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import os
import shutil

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def clean_cpp_headers():
    header_only = os.path.join(CURRENT_DIR, "..", "awkward-cpp", "header-only")
    if os.path.exists(header_only):
        shutil.rmtree(header_only)


if __name__ == "__main__":
    clean_cpp_headers()
