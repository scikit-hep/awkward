# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("to_cudf",)


@high_level_function()
def to_cudf(array):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).

    Converts an Awkward Array into a cuDF Series.

    Buffers that are not already in GPU memory will be transferred, and some
    structural reformatting may happen to account for differences in architecture.

    This function requires the `cudf` library (< 25.12.00) and a compatible GPU.
    cuDF versions 25.12.00 and later are not currently supported due to
    incompatible changes in cuDF internals.

    See also #ak.to_cupy, #ak.from_cupy, #ak.to_dataframe.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array)


def _impl(array):
    try:
        import cudf
    except ImportError as err:
        raise ImportError(
            """to use ak.to_cudf, you must install the 'cudf' package with:

    pip install cudf-cu13
or
    conda install -c rapidsai cudf cuda-version=13"""
        ) from err

    from packaging.version import parse as parse_version

    if parse_version(cudf.__version__) >= parse_version("25.12.00"):
        raise NotImplementedError(
            f"ak.to_cudf is not supported for cudf >= 25.12.00 (you have {cudf.__version__}). "
            "cudf internals changed in ways that are incompatible with the current implementation"
        )

    layout = ak.to_layout(array, allow_record=False)

    if hasattr(cudf.Series, "_from_column"):
        return cudf.Series._from_column(layout._to_cudf(cudf, None, len(layout)))
    # older Series invocation
    return cudf.Series(layout._to_cudf(cudf, None, len(layout)))
