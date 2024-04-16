# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from awkward._dispatch import high_level_function

__all__ = ("to_cudf",)


@high_level_function()
def to_cudf(
    array,
):
    import cudf
    return cudf.Series(array.layout._to_cudf(cudf, None, len(array)))
