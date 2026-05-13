# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout
from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward.contents import (
    ByteMaskedArray,
    IndexedArray,
    ListOffsetArray,
    NumpyArray,
    RecordArray,
)
from awkward.index import Index8, Index32

try:
    import pylibcudf as _pylibcudf
except ModuleNotFoundError:
    _pylibcudf = None
    _HAVE_PYLIBCUDF = False
else:
    _HAVE_PYLIBCUDF = True


__all__ = ("from_cudf",)


np = NumpyMetadata.instance()
_ISSUE_URL = "https://github.com/scikit-hep/awkward/issues"


@high_level_function()
def from_cudf(obj, *, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        obj (cudf.Series or cudf.DataFrame): The cuDF object to convert into an
            Awkward Array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Converts a cuDF Series or DataFrame into an Awkward Array by recursively
    traversing pylibcudf columns and wrapping GPU buffers with CuPy.

    See also #ak.to_cudf, #ak.from_cupy, and #ak.from_dlpack.
    """
    # Dispatch
    yield (obj,)

    # Implementation
    return _impl(obj, highlevel, behavior, attrs)


def _cupy():
    try:
        return Cupy.instance()._module
    except ModuleNotFoundError as err:
        raise ImportError(
            """to use ak.from_cudf, you must install the 'cupy' package with:

    pip install cupy-cuda12x
or
    conda install -c conda-forge cupy"""
        ) from err


def _require_pylibcudf():
    if not _HAVE_PYLIBCUDF:
        raise ImportError(
            """to use ak.from_cudf, you must install the 'pylibcudf' package.

It is distributed with cuDF. For example:

    pip install cudf-cu12
or
    conda install -c rapidsai cudf"""
        )

    return _pylibcudf


def _type_id_class(plc):
    return getattr(plc, "TypeId", None) or plc.types.TypeId


def _type_id(plc, name):
    return getattr(_type_id_class(plc), name, None)


def _column_size(col):
    size = col.size
    return size() if callable(size) else size


def _column_offset(col):
    offset = getattr(col, "offset", 0)
    return offset() if callable(offset) else offset


def _column_null_count(col):
    null_count = col.null_count
    return null_count() if callable(null_count) else null_count


def _num_children(col):
    num_children = getattr(col, "num_children", None)
    if callable(num_children):
        return num_children()
    elif num_children is not None:
        return num_children
    else:
        return len(col.children())


def _child_names(col):
    num_children = _num_children(col)
    child_names = getattr(col, "child_names", None)

    if callable(child_names):
        names = child_names()
    elif child_names is None:
        names = None
    else:
        names = child_names

    if names is None:
        return [str(i) for i in range(num_children)]

    names = list(names)
    if len(names) < num_children:
        names.extend(str(i) for i in range(len(names), num_children))

    return names[:num_children]


def _to_pylibcudf_column(series):
    try:
        result = series.to_pylibcudf()
    except AttributeError as err:
        raise RuntimeError(
            "cudf.Series.to_pylibcudf() is required by ak.from_cudf. "
            "Please use cudf >= 25.02."
        ) from err

    return result[0] if isinstance(result, tuple) else result


def _buf_to_cupy(buf, dtype: str):
    """
    Wrap a pylibcudf gpumemoryview as a CuPy array without copying.

    Missing buffers represent empty columns in pylibcudf, so they become
    zero-length CuPy arrays of the requested dtype.
    """
    cp = _cupy()

    if buf is None:
        return cp.empty(0, dtype=dtype)

    dtype = np.dtype(dtype)
    mem = cp.cuda.UnownedMemory(ptr=int(buf.ptr), size=int(buf.size), owner=buf)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    length = int(buf.size) // dtype.itemsize
    return cp.ndarray(length, dtype=dtype, memptr=memptr)


def _data_to_cupy(col, dtype):
    data = _buf_to_cupy(col.data_buffer(), dtype)

    if data.size == 0 and _column_size(col) != 0:
        if _column_null_count(col) == _column_size(col):
            return _cupy().empty(_column_offset(col) + _column_size(col), dtype=dtype)
        raise RuntimeError("non-empty pylibcudf column is missing its data buffer")

    return data


def _offsets_to_index(offsets_col, parent_col):
    cp = _cupy()
    fallback_length = max(_column_offset(parent_col) + _column_size(parent_col) + 1, 1)

    if offsets_col is None:
        offsets = cp.zeros(fallback_length, dtype="int32")
    else:
        offsets = _buf_to_cupy(offsets_col.data_buffer(), "int32")
        if offsets.size == 0:
            offsets = cp.zeros(fallback_length, dtype="int32")

    return Index32(offsets)


def _unpack_null_mask(mask, size, offset):
    cp = _cupy()

    if mask is None or size == 0:
        return None

    bitmask = _buf_to_cupy(mask, "uint8")
    if bitmask.size == 0:
        return None

    bit_offset = offset if bitmask.size * 8 >= offset + size else 0
    positions = cp.arange(size, dtype=cp.int64) + bit_offset
    byte_mask = (bitmask[positions >> 3] >> (positions & 7)) & 1
    return byte_mask.astype(cp.int8)


def _finalize(layout, col):
    """
    Apply the logical column offset and nullable mask to a layout.

    libcudf uses Arrow-style packed validity bits, where 1 means valid.
    Awkward's ByteMaskedArray uses one byte per element with the same
    ``valid_when=True`` convention after unpacking.
    """
    offset = _column_offset(col)
    size = _column_size(col)
    stop = offset + size

    if layout.length != size:
        if layout.length < stop:
            raise RuntimeError(
                "pylibcudf column buffers have an unexpected shape for ak.from_cudf"
            )
        layout = layout[offset:stop]

    if _column_null_count(col) != 0:
        byte_mask = _unpack_null_mask(col.null_mask(), size, offset)
        if byte_mask is not None:
            layout = ByteMaskedArray.simplified(
                Index8(byte_mask),
                layout,
                valid_when=True,
            )

    return layout


def _primitive_dtypes(plc):
    names_and_dtypes = (
        ("INT8", "int8"),
        ("INT16", "int16"),
        ("INT32", "int32"),
        ("INT64", "int64"),
        ("UINT8", "uint8"),
        ("UINT16", "uint16"),
        ("UINT32", "uint32"),
        ("UINT64", "uint64"),
        ("FLOAT32", "float32"),
        ("FLOAT64", "float64"),
        ("BOOL8", "bool"),
        ("DURATION_DAYS", "int64"),
        ("DURATION_SECONDS", "int64"),
        ("DURATION_MILLISECONDS", "int64"),
        ("DURATION_MICROSECONDS", "int64"),
        ("DURATION_NANOSECONDS", "int64"),
        ("TIMESTAMP_DAYS", "int64"),
        ("TIMESTAMP_SECONDS", "int64"),
        ("TIMESTAMP_MILLISECONDS", "int64"),
        ("TIMESTAMP_MICROSECONDS", "int64"),
        ("TIMESTAMP_NANOSECONDS", "int64"),
    )

    out = {}
    for name, dtype in names_and_dtypes:
        type_id = _type_id(plc, name)
        if type_id is not None:
            out[type_id] = dtype
    return out


def _column_to_layout(col):
    plc = _require_pylibcudf()
    type_id = col.type().id()
    primitive_dtypes = _primitive_dtypes(plc)

    if type_id in primitive_dtypes:
        layout = NumpyArray(_data_to_cupy(col, primitive_dtypes[type_id]))

    elif type_id == _type_id(plc, "LIST"):
        layout = ListOffsetArray(
            _offsets_to_index(col.child(0), col),
            _column_to_layout(col.child(1)),
        )

    elif type_id == _type_id(plc, "STRUCT"):
        layout = RecordArray(
            [_column_to_layout(col.child(i)) for i in range(_num_children(col))],
            _child_names(col),
            length=_column_size(col),
        )

    elif type_id == _type_id(plc, "STRING"):
        chars_col = col.child(1)
        layout = ListOffsetArray(
            _offsets_to_index(col.child(0), col),
            NumpyArray(
                _data_to_cupy(chars_col, "uint8"),
                parameters={"__array__": "char"},
            ),
            parameters={"__array__": "string"},
        )

    elif type_id == _type_id(plc, "DICTIONARY32"):
        layout = IndexedArray(
            Index32(_data_to_cupy(col.child(0), "int32")),
            _column_to_layout(col.child(1)),
        )

    else:
        raise NotImplementedError(
            f"pylibcudf type id {type_id!r} is not supported by ak.from_cudf yet. "
            f"Please open an issue at {_ISSUE_URL} if you need this type."
        )

    return _finalize(layout, col)


def _dataframe_to_layout(dataframe):
    fields = list(dataframe.columns)
    contents = [
        _column_to_layout(_to_pylibcudf_column(dataframe[name])) for name in fields
    ]
    return RecordArray(contents, fields, length=len(dataframe))


def _impl(obj, highlevel, behavior, attrs):
    try:
        import cudf
    except ModuleNotFoundError as err:
        raise ImportError(
            """to use ak.from_cudf, you must install the 'cudf' package with:

    pip install cudf-cu12
or
    conda install -c rapidsai cudf"""
        ) from err

    if isinstance(obj, cudf.Series):
        layout = _column_to_layout(_to_pylibcudf_column(obj))
    elif isinstance(obj, cudf.DataFrame):
        layout = _dataframe_to_layout(obj)
    else:
        raise TypeError(
            "ak.from_cudf accepts only cudf.Series or cudf.DataFrame, "
            f"not {type(obj).__name__!r}"
        )

    return wrap_layout(
        layout,
        highlevel=highlevel,
        behavior=behavior,
        attrs=attrs,
    )
