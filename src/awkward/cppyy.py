# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

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

conda install -c conda-forge cppyy"""
        ) from err

    if not _has_checked_version:
        if ak._util.parse_version(cppyy.__version__) < ak._util.parse_version("3.0.1"):
            raise ImportError(
                "Awkward Array can only work with cppyy 3.0.1 or later "
                "(you have version {})".format(cppyy.__version__)
            )
        _has_checked_version = True
