# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("to_cudf",)


@high_level_function()
def to_cudf(
    array: ak.Array,
):
    """Create a cuDF.Series out of the given ak array

    Buffers that are not already in GPU memory will be transferred, and some
    structural reformatting may happen to account for differences in architecture.
    """
    import cudf

    if hasattr(cudf.Series, "_from_column"):
        return cudf.Series._from_column(array.layout._to_cudf(cudf, None, len(array)))
    # older Series invocation
    return cudf.Series(array.layout._to_cudf(cudf, None, len(array)))
