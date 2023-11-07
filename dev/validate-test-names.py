#!/usr/bin/env python3

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import argparse
import pathlib
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=pathlib.Path, nargs="*")
    args = parser.parse_args()

    for path in args.paths:
        if not re.match(r"^[^\d\W]\w*\Z$", path.stem):
            raise ValueError(
                f"tests must be named according to the pattern "
                f"'test_XXXX_some_identifier_here.py'. {path.stem} is "
                f"not named correctly."
            )
