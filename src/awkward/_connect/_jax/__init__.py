# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# v2: replace with deeply rewritten src/awkward/_v2/_connect/jax.

import types

import awkward as ak

checked_version = False


def register_and_check():
    global checked_version
    try:
        import jax
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            """install the 'jax' package with:

    pip install jax jaxlib --upgrade

or

    conda install jax jaxlib"""
        ) from None
    else:
        if not checked_version and ak._v2._util.parse_version(
            jax.__version__
        ) < ak._v2._util.parse_version("0.2.7"):
            raise ImportError(
                "Awkward Array can only work with jax 0.2.7 or later "
                "(you have version {})".format(jax.__version__)
            )
        checked_version = True
        register()


def register():
    import awkward._connect._jax.jax_utils  # noqa: F401


ak.jax = types.ModuleType("jax")
ak.jax.register = register_and_check
