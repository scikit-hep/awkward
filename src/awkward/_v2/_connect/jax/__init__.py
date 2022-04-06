# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

try:
    import jax

    error_message = None

except ModuleNotFoundError:
    jax = None
    error_message = """to use {0}, you must install jax:

    pip install jax jaxlib

or

    conda install -c conda-forge jax jaxlib
"""


def import_jax(name="Awkward Arrays with JAX"):
    if jax is None:
        raise ImportError(error_message.format(name))
    return jax


def register_jax():
    import awkward as ak
    from awkward._v2._connect.jax.nplike import Jax

    # JAX Specific Configurations
    jax.config.update("jax_enable_x64", True)

    ak._v2._util._backends["jax"] = Jax
