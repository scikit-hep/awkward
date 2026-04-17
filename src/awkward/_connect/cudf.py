# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward.contents import NumpyArray

__all__ = ("from_cudf",)

# Maps Arrow C Data Interface format strings to NumPy dtype strings.
# Booleans ("b") are intentionally omitted: Arrow booleans are bit-packed, so
# the fixed-width numeric path through buffers[1] would produce incorrect data.
# See https://arrow.apache.org/docs/format/CDataInterface.html#data-type-description-format-strings
_ARROW_FORMAT_TO_DTYPE: dict[str, str] = {
    "c": "int8",
    "C": "uint8",
    "s": "int16",
    "S": "uint16",
    "i": "int32",
    "I": "uint32",
    "l": "int64",
    "L": "uint64",
    "e": "float16",
    "f": "float32",
    "g": "float64",
}


def _primitive_dtype_from_arrow_format(arrow_format):
    try:
        return _ARROW_FORMAT_TO_DTYPE[arrow_format]
    except KeyError as err:
        raise NotImplementedError(
            f"Arrow format {arrow_format!r} is not supported by ak.from_cudf. "
            "Only flat, fixed-width numeric cudf.Series are currently supported; "
            "nulls, booleans, strings, categorical data, and nested types are "
            "not supported."
        ) from err


def _to_pylibcudf_column(series):
    try:
        result = series.to_pylibcudf()
    except AttributeError as err:
        raise RuntimeError(
            "cudf.Series.to_pylibcudf() is required by ak.from_cudf. "
            "Please use cudf >= 25.02."
        ) from err

    # to_pylibcudf() may return a bare Column or a (Column, metadata) tuple
    # depending on the cuDF version.
    return result[0] if isinstance(result, tuple) else result


def from_cudf(series):
    """
    Args:
        series (cudf.Series): The cuDF Series to convert into a low-level
            Awkward layout.

    Converts a flat cuDF Series into a low-level #ak.contents.NumpyArray using
    the Arrow C Device Interface.

    Only flat, fixed-width numeric dtypes are supported in this initial
    implementation. Null values, booleans, strings, categorical data, and
    nested types raise ``NotImplementedError``.
    """

    try:
        import cudf
    except ImportError as err:
        raise ImportError(
            """to use ak.from_cudf, you must install the 'cudf' package with:

    pip install cudf-cu13
or
    conda install -c rapidsai cudf cuda-version=13"""
        ) from err

    if not isinstance(series, cudf.Series):
        raise TypeError(
            f"ak.from_cudf accepts only cudf.Series, not {type(series).__name__!r}"
        )

    #  Obtain the pylibcudf Column via the public API (cuDF >= 25.02)
    plc_column = _to_pylibcudf_column(series)

    # Verify the Arrow C Device Interface is present on this object

    if not hasattr(plc_column, "__arrow_c_device_array__"):
        raise RuntimeError("pylibcudf column does not support Arrow device interface")

    # Import nanoarrow (required to consume the PyCapsule protocol)

    try:
        import nanoarrow.device as nanoarrow_device
    except ImportError as err:
        raise ImportError(
            """to use ak.from_cudf, you must install the 'nanoarrow' package with:

    pip install nanoarrow"""
        ) from err

    try:
        # nanoarrow.device.c_device_array() consumes the public
        # __arrow_c_device_array__() protocol — no data is copied.
        device_array = nanoarrow_device.c_device_array(plc_column)
    except (TypeError, ValueError, RuntimeError) as err:
        raise RuntimeError(
            "failed to build a nanoarrow device array from the pylibcudf column"
        ) from err

    if device_array.null_count > 0:
        raise NotImplementedError(
            "ak.from_cudf does not yet support cudf.Series with null values. "
            "Use series.dropna() or series.fillna() before converting."
        )

    # Resolve dtype from the Arrow schema format string
    # device_array.schema.format is  Arrow C Data Interface format string,
    # rejected because Arrow booleans are bit-packed — treating their values
    # buffer as fixed-width bytes would silently produce incorrect results.
    dtype = _primitive_dtype_from_arrow_format(device_array.schema.format)

    #  CuPy

    try:
        import cupy as cp
    except ImportError as err:
        raise ImportError(
            """to use ak.from_cudf, you must install the 'cupy' package with:

    pip install cupy-cuda13x"""
        ) from err

    # Extract the values buffer and build a CuPy array
    # Arrow primitive buffer layout:
    #   buffers[0] -> validity bitmap (ignored — null_count == 0 is enforced above)
    #   buffers[1] -> values buffer
    buffers = device_array.buffers
    if len(buffers) < 2:
        raise RuntimeError(
            "The exported Arrow device array does not have the primitive values "
            "buffer expected by ak.from_cudf."
        )

    data_buffer = buffers[1]

    if data_buffer is None:
        # Arrow permits NULL buffers for empty arrays.  There is no device
        # memory to share, so construct an empty CuPy array with the right dtype.
        if device_array.length == 0:
            return NumpyArray(cp.asarray((), dtype=dtype))

        raise RuntimeError(
            "The exported Arrow device array is missing its values buffer."
        )

    # cp.asarray() honours the __cuda_array_interface__ exposed by nanoarrow
    # buffer objects — no host copy occurs.
    data = cp.asarray(data_buffer, dtype=dtype)

    # 10. Apply Arrow offset (e.g. from a sliced cuDF Series)

    # Arrow allows a non-zero offset into the backing allocation so that
    # slices share memory with the parent.  We must respect it.
    if device_array.offset != 0 or data.shape[0] != device_array.length:
        start = device_array.offset
        stop = start + device_array.length

        if data.shape[0] < stop:
            raise RuntimeError(
                "The exported Arrow device array has an unexpected values buffer "
                "shape for ak.from_cudf."
            )

        data = data[start:stop]

    return NumpyArray(data)
