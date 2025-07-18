# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations


def register_and_check():
    """
    Register Awkward Array node types with JAX's tree mechanism.
    """
    try:
        import torch
        import awkward._connect.torch

    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            """install the 'torch' package with:

        python3 -m pip install torch

    or

        conda install -c conda-forge torch
    """
        ) from None


def import_torch():
    """Ensure that Torch integration is registered, and return the Torch module. Raise a RuntimeError if not."""
    import torch
    return torch