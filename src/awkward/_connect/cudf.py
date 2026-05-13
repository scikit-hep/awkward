# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward.contents import BitMaskedArray, ListOffsetArray, NumpyArray, RecordArray
from awkward.index import Index32, Index64, IndexU8

__all__ = ("from_cudf",)


def _to_pylibcudf_column(series):
    try:
        result = series.to_pylibcudf()
    except AttributeError as err:
        raise RuntimeError(
            "cudf.Series.to_pylibcudf() is required by ak.from_cudf. "
            "Please use cudf >= 25.02."
        ) from err

    return result[0] if isinstance(result, tuple) else result


def from_cudf(series):
    """
    Args:
        series (cudf.Series): The cuDF Series to convert into a low-level
            Awkward layout.

    Converts a cuDF Series into a low-level Awkward layout by recursively
    traversing the corresponding pylibcudf column.

    Primitive, boolean, list, struct, string, and nullable columns are
    supported. Other column types raise ``NotImplementedError``.
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

    try:
        import cupy as cp
    except ImportError as err:
        raise ImportError(
            """to use ak.from_cudf, you must install the 'cupy' package with:

    pip install cupy-cuda13x"""
        ) from err

    try:
        import pylibcudf as plc
    except ImportError as err:
        raise ImportError(
            """to use ak.from_cudf, you must install the 'pylibcudf' package with:

    pip install pylibcudf-cu13"""
        ) from err

    if not isinstance(series, cudf.Series):
        raise TypeError(
            f"ak.from_cudf accepts only cudf.Series, not {type(series).__name__!r}"
        )

    plc_column = _to_pylibcudf_column(series)

    primitive_type_id_to_dtype = {
        plc.types.TypeId.INT8: "int8",
        plc.types.TypeId.INT16: "int16",
        plc.types.TypeId.INT32: "int32",
        plc.types.TypeId.INT64: "int64",
        plc.types.TypeId.UINT8: "uint8",
        plc.types.TypeId.UINT16: "uint16",
        plc.types.TypeId.UINT32: "uint32",
        plc.types.TypeId.UINT64: "uint64",
        plc.types.TypeId.FLOAT32: "float32",
        plc.types.TypeId.FLOAT64: "float64",
        plc.types.TypeId.TIMESTAMP_DAYS: "datetime64[D]",
        plc.types.TypeId.TIMESTAMP_SECONDS: "datetime64[s]",
        plc.types.TypeId.TIMESTAMP_MILLISECONDS: "datetime64[ms]",
        plc.types.TypeId.TIMESTAMP_MICROSECONDS: "datetime64[us]",
        plc.types.TypeId.TIMESTAMP_NANOSECONDS: "datetime64[ns]",
        plc.types.TypeId.DURATION_DAYS: "timedelta64[D]",
        plc.types.TypeId.DURATION_SECONDS: "timedelta64[s]",
        plc.types.TypeId.DURATION_MILLISECONDS: "timedelta64[ms]",
        plc.types.TypeId.DURATION_MICROSECONDS: "timedelta64[us]",
        plc.types.TypeId.DURATION_NANOSECONDS: "timedelta64[ns]",
    }

    def _empty_array(dtype):
        return cp.asarray((), dtype=dtype)

    def _column_size(col):
        return col.size() if callable(col.size) else col.size

    def _column_offset(col):
        return col.offset() if callable(col.offset) else col.offset

    def _column_null_count(col):
        return col.null_count() if callable(col.null_count) else col.null_count

    def _asarray(buffer, dtype=None):
        data = cp.asarray(buffer)

        if dtype is None:
            return data

        target_dtype = cp.dtype(dtype)
        if data.dtype == target_dtype:
            return data

        if data.dtype.itemsize == target_dtype.itemsize:
            return data.view(target_dtype)

        return cp.asarray(buffer, dtype=target_dtype)

    def _data_to_cupy(col, dtype):
        buffer = col.data_buffer()

        if buffer is None:
            if _column_size(col) == 0:
                return _empty_array(dtype)
            raise RuntimeError("non-empty pylibcudf column is missing its data buffer")

        return _asarray(buffer, dtype=dtype)

    def _offsets_to_index(offsets_source):
        if offsets_source is None:
            offsets = _empty_array("int64")
        else:
            buffer = offsets_source.data_buffer()
            if buffer is None:
                if _column_size(offsets_source) == 0:
                    offsets = _empty_array("int64")
                else:
                    raise RuntimeError(
                        "non-empty pylibcudf offsets column is missing its data buffer"
                    )
            else:
                offsets = _asarray(buffer)

        if offsets.shape[0] == 0:
            offsets = cp.asarray((0,), dtype="int64")
            return Index64(offsets)

        if offsets.dtype == cp.dtype("int32"):
            return Index32(offsets)
        elif offsets.dtype == cp.dtype("int64"):
            return Index64(offsets)
        else:
            return Index64(cp.asarray(offsets, dtype="int64"))

    def _struct_fields(col, dtype, num_children):
        fields_method = getattr(col.type(), "fields", None)
        if callable(fields_method):
            fields = list(fields_method())
        elif fields_method is None:
            fields = []
        else:
            fields = list(fields_method)

        names = []
        for i, field in enumerate(fields[:num_children]):
            field_name = getattr(field, "name", None)
            if callable(field_name):
                field_name = field_name()
            if field_name is None and isinstance(field, tuple) and len(field) > 0:
                field_name = field[0]
            names.append(str(i) if field_name is None else field_name)

        child_dtypes = []
        dtype_fields = getattr(dtype, "fields", None)
        if dtype_fields is not None:
            child_dtypes.extend(child_dtype for _, child_dtype in dtype_fields.items())

            if len(names) < num_children:
                for name in dtype_fields:
                    if len(names) == num_children:
                        break
                    if name not in names:
                        names.append(name)

        if len(names) < num_children:
            names.extend(str(i) for i in range(len(names), num_children))

        if len(child_dtypes) < num_children:
            child_dtypes.extend([None] * (num_children - len(child_dtypes)))

        return names, child_dtypes[:num_children]

    def _finalize(layout, col):
        offset = _column_offset(col)
        size = _column_size(col)
        stop = offset + size

        if layout.length < stop:
            raise RuntimeError(
                "pylibcudf column buffers have an unexpected shape for ak.from_cudf"
            )

        if _column_null_count(col) != 0:
            mask = col.null_mask()
            if mask is not None:
                layout = BitMaskedArray.simplified(
                    IndexU8(cp.asarray(mask, dtype=cp.uint8)),
                    layout,
                    valid_when=True,
                    length=stop,
                    lsb_order=True,
                )

        if offset != 0 or layout.length != size:
            layout = layout[offset:stop]

        return layout

    def recurse(col, dtype=None):
        type_id = col.type().id()

        if type_id in primitive_type_id_to_dtype:
            layout = NumpyArray(_data_to_cupy(col, primitive_type_id_to_dtype[type_id]))

        elif type_id == plc.types.TypeId.BOOL8:
            # cuDF stores bool as bytes, so a cast is required here.
            layout = NumpyArray(_data_to_cupy(col, cp.uint8).astype(cp.bool_))

        elif type_id == plc.types.TypeId.LIST:
            offsets_source = col.offsets()
            child_col = col.child(0)
            child_dtype = getattr(dtype, "element_type", None)
            layout = ListOffsetArray(
                _offsets_to_index(offsets_source),
                recurse(child_col, child_dtype),
            )

        elif type_id == plc.types.TypeId.STRUCT:
            num_children = len(col.children())
            field_names, child_dtypes = _struct_fields(col, dtype, num_children)
            contents = [
                recurse(col.child(i), child_dtypes[i]) for i in range(num_children)
            ]
            layout = RecordArray(
                contents,
                field_names,
                length=_column_offset(col) + _column_size(col),
            )

        elif type_id == plc.types.TypeId.STRING:
            offsets_col = col.child(0)
            chars_buffer = col.data_buffer()
            layout = ListOffsetArray(
                _offsets_to_index(offsets_col),
                NumpyArray(
                    _empty_array(cp.uint8)
                    if chars_buffer is None
                    else _asarray(chars_buffer, dtype=cp.uint8),
                    parameters={"__array__": "char"},
                ),
                parameters={"__array__": "string"},
            )

        else:
            raise NotImplementedError(
                f"pylibcudf type id {type_id!r} is not supported by ak.from_cudf"
            )

        return _finalize(layout, col)

    return recurse(plc_column, series.dtype)
