# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import warnings

import awkward as ak

checked_version = False


def import_tensorflow():
    global checked_version
    try:
        import tensorflow
    except ModuleNotFoundError as err:
        raise ak._v2._util.error(
            ModuleNotFoundError(
                """install the 'tensorflow' package with:

    pip install tensorflow --upgrade

or

    conda install tensorflow"""
            )
        ) from err
    else:
        if not checked_version and ak._v2._util.parse_version(
            tensorflow.__version__
        ) < ak._v2._util.parse_version("1.15.0"):
            warnings.warn(
                "Awkward Array is only known to work with tensorflow 1.15.0 or later"
                "(you have version {})".format(tensorflow.__version__),
                RuntimeWarning,
            )
        checked_version = True
        return tensorflow
