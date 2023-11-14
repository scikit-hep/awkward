# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import packaging.version

_has_checked_version = False


def register_and_check():
    global _has_checked_version

    try:
        import cppyy
    except ImportError as err:
        raise ImportError(
            """install the 'cppyy' package with:

pip install cppyy

or

conda install -c conda-forge cppyy

Note that this must be in a different venv or conda environment from ROOT, if you have installed ROOT.
"""
        ) from err

    if not _has_checked_version:
        if packaging.version.Version(cppyy.__version__) < packaging.version.Version(
            "3.1.0"
        ):
            raise ImportError(
                "Awkward Array can only work with cppyy 3.1.0 or later "
                f"(you have version {cppyy.__version__})"
            )
        _has_checked_version = True
