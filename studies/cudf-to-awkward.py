import cudf
import pyarrow
import cupy
import numpy

import awkward as ak
from awkward._backends.numpy import NumpyBackend
from awkward._backends.cupy import CupyBackend


######################### stripped-down copy of src/awkward/_connect/pyarrow.py


_string_like = (
    pyarrow.string(),
    pyarrow.large_string(),
    pyarrow.binary(),
    pyarrow.large_binary(),
)

_pyarrow_to_numpy_dtype = {
    pyarrow.date32(): (True, numpy.dtype("M8[D]")),
    pyarrow.date64(): (False, numpy.dtype("M8[ms]")),
    pyarrow.time32("s"): (True, numpy.dtype("M8[s]")),
    pyarrow.time32("ms"): (True, numpy.dtype("M8[ms]")),
    pyarrow.time64("us"): (False, numpy.dtype("M8[us]")),
    pyarrow.time64("ns"): (False, numpy.dtype("M8[ns]")),
    pyarrow.timestamp("s"): (False, numpy.dtype("M8[s]")),
    pyarrow.timestamp("ms"): (False, numpy.dtype("M8[ms]")),
    pyarrow.timestamp("us"): (False, numpy.dtype("M8[us]")),
    pyarrow.timestamp("ns"): (False, numpy.dtype("M8[ns]")),
    pyarrow.duration("s"): (False, numpy.dtype("m8[s]")),
    pyarrow.duration("ms"): (False, numpy.dtype("m8[ms]")),
    pyarrow.duration("us"): (False, numpy.dtype("m8[us]")),
    pyarrow.duration("ns"): (False, numpy.dtype("m8[ns]")),
}


def revertable(modified, original):
    modified.__pyarrow_original = original
    return modified


def remove_optiontype(akarray):
    return akarray.__pyarrow_original


def popbuffers_finalize(out, array, validbits, generate_bitmasks, fix_offsets=True):
    # Every buffer from Arrow must be offsets-corrected.
    if fix_offsets and (array.offset != 0 or len(array) != len(out)):
        out = out[array.offset : array.offset + len(array)]

    # Everything must leave popbuffers as option-type; the mask_node will be
    # removed by the next level up in popbuffers recursion if appropriate.

    if validbits is None and generate_bitmasks:
        # ceildiv(len(out), 8) = -(len(out) // -8)
        validbits = numpy.full(-(len(out) // -8), numpy.uint8(0xFF), dtype=numpy.uint8)

    if validbits is None:
        return revertable(ak.contents.UnmaskedArray.simplified(out), out)
    else:
        return revertable(
            ak.contents.BitMaskedArray.simplified(
                ak.index.IndexU8(numpy.frombuffer(validbits, dtype=numpy.uint8)),
                out,
                valid_when=True,
                length=len(out),
                lsb_order=True,
            ),
            out,
        )


def popbuffers(paarray, arrow_type, buffers, generate_bitmasks):
    ### Beginning of the big if-elif-elif chain!

    if isinstance(arrow_type, pyarrow.lib.DictionaryType):
        masked_index = popbuffers(
            paarray.indices,
            arrow_type.index_type,
            buffers,
            generate_bitmasks,
        )
        index = masked_index.content.data

        if not isinstance(masked_index, ak.contents.UnmaskedArray):
            mask = masked_index.mask_as_bool(valid_when=False)
            if mask.any():
                index = numpy.asarray(index, copy=True)
                index[mask] = -1

        content = handle_arrow(paarray.dictionary, generate_bitmasks)

        parameters = {"__array__": "categorical"}

        return revertable(
            ak.contents.IndexedOptionArray.simplified(
                ak.index.Index(index),
                content,
                parameters=parameters,
            ),
            ak.contents.IndexedArray(
                ak.index.Index(index),
                remove_optiontype(content) if content.is_option else content,
                parameters=parameters,
            ),
        )

    elif isinstance(arrow_type, pyarrow.lib.FixedSizeListType):
        assert arrow_type.num_buffers == 1
        validbits = buffers.pop(0)

        akcontent = popbuffers(
            paarray.values, arrow_type.value_type, buffers, generate_bitmasks
        )

        if not arrow_type.value_field.nullable:
            # strip the dummy option-type node
            akcontent = remove_optiontype(akcontent)

        out = ak.contents.RegularArray(
            akcontent,
            arrow_type.list_size,
            parameters=None,
        )
        return popbuffers_finalize(out, paarray, validbits, generate_bitmasks)

    elif isinstance(arrow_type, (pyarrow.lib.LargeListType, pyarrow.lib.ListType)):
        assert arrow_type.num_buffers == 2
        validbits = buffers.pop(0)
        paoffsets = buffers.pop(0)

        if isinstance(arrow_type, pyarrow.lib.LargeListType):
            akoffsets = ak.index.Index64(numpy.frombuffer(paoffsets, dtype=numpy.int64))
        else:
            akoffsets = ak.index.Index32(numpy.frombuffer(paoffsets, dtype=numpy.int32))

        akcontent = popbuffers(
            paarray.values, arrow_type.value_type, buffers, generate_bitmasks
        )

        if not arrow_type.value_field.nullable:
            # strip the dummy option-type node
            akcontent = remove_optiontype(akcontent)

        out = ak.contents.ListOffsetArray(akoffsets, akcontent, parameters=None)
        return popbuffers_finalize(out, paarray, validbits, generate_bitmasks)

    elif isinstance(arrow_type, pyarrow.lib.MapType):
        # FIXME: make a ListOffsetArray of 2-tuples with __array__ == "sorted_map".
        # (Make sure the keys are sorted).
        raise NotImplementedError

    elif isinstance(
        arrow_type, (pyarrow.lib.Decimal128Type, pyarrow.lib.Decimal256Type)
    ):
        # Note: Decimal128Type and Decimal256Type are subtypes of FixedSizeBinaryType.
        # NumPy doesn't support decimal: https://github.com/numpy/numpy/issues/9789
        raise ValueError(
            "Arrow arrays containing pyarrow.decimal128 or pyarrow.decimal256 types can't be converted into Awkward Arrays"
        )

    elif isinstance(arrow_type, pyarrow.lib.FixedSizeBinaryType):
        assert arrow_type.num_buffers == 2
        validbits = buffers.pop(0)
        pacontent = buffers.pop(0)

        parameters = {"__array__": "bytestring"}
        sub_parameters = {"__array__": "byte"}

        out = ak.contents.RegularArray(
            ak.contents.NumpyArray(
                numpy.frombuffer(pacontent, dtype=numpy.uint8),
                parameters=sub_parameters,
                backend=NumpyBackend.instance(),
            ),
            arrow_type.byte_width,
            parameters=parameters,
        )
        return popbuffers_finalize(out, paarray, validbits, generate_bitmasks)

    elif arrow_type in _string_like:
        assert arrow_type.num_buffers == 3
        validbits = buffers.pop(0)
        paoffsets = buffers.pop(0)
        pacontent = buffers.pop(0)

        if arrow_type in _string_like[::2]:
            akoffsets = ak.index.Index32(numpy.frombuffer(paoffsets, dtype=numpy.int32))
        else:
            akoffsets = ak.index.Index64(numpy.frombuffer(paoffsets, dtype=numpy.int64))

        if arrow_type in _string_like[:2]:
            parameters = {"__array__": "string"}
            sub_parameters = {"__array__": "char"}
        else:
            parameters = {"__array__": "bytestring"}
            sub_parameters = {"__array__": "byte"}

        out = ak.contents.ListOffsetArray(
            akoffsets,
            ak.contents.NumpyArray(
                numpy.frombuffer(pacontent, dtype=numpy.uint8),
                parameters=sub_parameters,
                backend=NumpyBackend.instance(),
            ),
            parameters=parameters,
        )
        return popbuffers_finalize(out, paarray, validbits, generate_bitmasks)

    elif isinstance(arrow_type, pyarrow.lib.StructType):
        assert arrow_type.num_buffers == 1
        validbits = buffers.pop(0)

        keys = []
        contents = []
        for i in range(arrow_type.num_fields):
            field = arrow_type[i]
            field_name = field.name
            keys.append(field_name)

            akcontent = popbuffers(
                paarray.field(field_name), field.type, buffers, generate_bitmasks
            )
            if not field.nullable:
                # strip the dummy option-type node
                akcontent = remove_optiontype(akcontent)
            contents.append(akcontent)

        out = ak.contents.RecordArray(
            contents, keys, length=len(paarray), parameters=None
        )
        return popbuffers_finalize(
            out, paarray, validbits, generate_bitmasks, fix_offsets=False
        )

    elif isinstance(arrow_type, pyarrow.lib.UnionType):
        if isinstance(arrow_type, pyarrow.lib.SparseUnionType):
            assert arrow_type.num_buffers == 2
            validbits = buffers.pop(0)
            nptags = numpy.frombuffer(buffers.pop(0), dtype=numpy.int8)
            npindex = numpy.arange(len(nptags), dtype=numpy.int32)
        else:
            assert arrow_type.num_buffers == 3
            validbits = buffers.pop(0)
            nptags = numpy.frombuffer(buffers.pop(0), dtype=numpy.int8)
            npindex = numpy.frombuffer(buffers.pop(0), dtype=numpy.int32)

        akcontents = []
        for i in range(arrow_type.num_fields):
            field = arrow_type[i]
            akcontent = popbuffers(
                paarray.field(i), field.type, buffers, generate_bitmasks
            )

            if not field.nullable:
                # strip the dummy option-type node
                akcontent = remove_optiontype(akcontent)
            akcontents.append(akcontent)

        out = ak.contents.UnionArray.simplified(
            ak.index.Index8(nptags),
            ak.index.Index32(npindex),
            akcontents,
            parameters=None,
        )
        return popbuffers_finalize(out, paarray, None, generate_bitmasks)

    elif arrow_type == pyarrow.null():
        validbits = buffers.pop(0)
        assert arrow_type.num_fields == 0

        # This is already an option-type and offsets-corrected, so no popbuffers_finalize.
        return ak.contents.IndexedOptionArray(
            ak.index.Index64(numpy.full(len(paarray), -1, dtype=numpy.int64)),
            ak.contents.EmptyArray(parameters=None),
            parameters=None,
        )

    elif arrow_type == pyarrow.bool_():
        assert arrow_type.num_buffers == 2
        validbits = buffers.pop(0)
        bitdata = buffers.pop(0)

        bytedata = numpy.unpackbits(
            numpy.frombuffer(bitdata, dtype=numpy.uint8), bitorder="little"
        )

        out = ak.contents.NumpyArray(
            bytedata.view(numpy.bool_),
            parameters=None,
            backend=NumpyBackend.instance(),
        )
        return popbuffers_finalize(out, paarray, validbits, generate_bitmasks)

    elif isinstance(arrow_type, pyarrow.lib.DataType):
        assert arrow_type.num_buffers == 2
        validbits = buffers.pop(0)
        data = buffers.pop(0)

        to64, dt = _pyarrow_to_numpy_dtype.get(str(arrow_type), (False, None))
        if to64:
            data = numpy.astype(
                numpy.frombuffer(data, dtype=numpy.int32), dtype=numpy.int64
            )
        if dt is None:
            dt = arrow_type.to_pandas_dtype()

        out = ak.contents.NumpyArray(
            numpy.frombuffer(data, dtype=dt),
            parameters=None,
            backend=NumpyBackend.instance(),
        )
        return popbuffers_finalize(out, paarray, validbits, generate_bitmasks)

    else:
        raise TypeError(f"unrecognized Arrow array type: {arrow_type!r}")


def handle_arrow(obj, generate_bitmasks):
    buffers = obj.buffers()
    out = popbuffers(obj, obj.type, buffers, generate_bitmasks)
    assert len(buffers) == 0
    return out


def pyarrow_to_awkward(
    pyarrow_array: pyarrow.lib.Array,
    generate_bitmasks=False,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    ctx = ak._layout.HighLevelContext(behavior=behavior, attrs=attrs).finalize()

    out = handle_arrow(pyarrow_array, generate_bitmasks)
    if isinstance(out, ak.contents.UnmaskedArray):
        out = remove_optiontype(out)

    def remove_revertable(layout, **kwargs):
        if hasattr(layout, "__pyarrow_original"):
            del layout.__pyarrow_original

    ak._do.recursively_apply(out, remove_revertable)

    return ctx.wrap(out, highlevel=highlevel)


######################### equivalent for CuDF


def recurse_finalize(
    out: ak.contents.Content,
    column: cudf.core.column.column.ColumnBase,
    validbits: None | cudf.core.buffer.buffer.Buffer,
    generate_bitmasks: bool,
    fix_offsets: bool = True,
):
    # Every buffer from Arrow must be offsets-corrected.
    if fix_offsets and (column.offset != 0 or len(column) != len(out)):
        out = out[column.offset : column.offset + len(column)]

    if validbits is None:
        return revertable(ak.contents.UnmaskedArray.simplified(out), out)
    else:
        return revertable(
            ak.contents.BitMaskedArray.simplified(
                ak.index.IndexU8(cupy.asarray(validbits)),
                out,
                valid_when=True,
                length=len(out),
                lsb_order=True,
            ),
            out,
        )


def recurse(
    column: cudf.core.column.column.ColumnBase,
    arrow_type: pyarrow.lib.DataType,
    generate_bitmasks: bool,
):
    if isinstance(arrow_type, pyarrow.lib.DictionaryType):
        raise NotImplementedError

    elif isinstance(arrow_type, pyarrow.lib.FixedSizeListType):
        raise NotImplementedError

    elif isinstance(arrow_type, (pyarrow.lib.LargeListType, pyarrow.lib.ListType)):
        validbits = column.base_mask
        paoffsets = column.offsets.base_data

        if isinstance(arrow_type, pyarrow.lib.LargeListType):
            akoffsets = ak.index.Index64(cupy.asarray(paoffsets).view(cupy.int64))
        else:
            akoffsets = ak.index.Index32(cupy.asarray(paoffsets).view(cupy.int32))

        akcontent = recurse(
            column.base_children[-1], arrow_type.value_type, generate_bitmasks
        )

        if not arrow_type.value_field.nullable:
            # strip the dummy option-type node
            akcontent = remove_optiontype(akcontent)

        out = ak.contents.ListOffsetArray(akoffsets, akcontent, parameters=None)
        return recurse_finalize(out, column, validbits, generate_bitmasks)

    elif isinstance(arrow_type, pyarrow.lib.MapType):
        raise NotImplementedError

    elif isinstance(
        arrow_type, (pyarrow.lib.Decimal128Type, pyarrow.lib.Decimal256Type)
    ):
        # Note: Decimal128Type and Decimal256Type are subtypes of FixedSizeBinaryType.
        # NumPy doesn't support decimal: https://github.com/numpy/numpy/issues/9789
        raise ValueError(
            "Arrow arrays containing pyarrow.decimal128 or pyarrow.decimal256 types can't be converted into Awkward Arrays"
        )

    elif isinstance(arrow_type, pyarrow.lib.FixedSizeBinaryType):
        raise NotImplementedError

    elif arrow_type in _string_like:
        raise NotImplementedError

    elif isinstance(arrow_type, pyarrow.lib.StructType):
        validbits = column.base_mask

        keys = []
        contents = []
        for i in range(arrow_type.num_fields):
            field = arrow_type[i]
            field_name = field.name
            keys.append(field_name)

            akcontent = recurse(column.base_children[i], field.type, generate_bitmasks)
            if not field.nullable:
                # strip the dummy option-type node
                akcontent = remove_optiontype(akcontent)
            contents.append(akcontent)

        out = ak.contents.RecordArray(
            contents, keys, length=len(column), parameters=None
        )
        return recurse_finalize(out, column, validbits, generate_bitmasks)

    elif isinstance(arrow_type, pyarrow.lib.UnionType):
        raise NotImplementedError

    elif arrow_type == pyarrow.null():
        raise NotImplementedError

    elif arrow_type == pyarrow.bool_():
        validbits = column.base_mask

        ## boolean data from CuDF differs from Arrow: it's represented as bytes, not bits!
        # bitdata = column.base_data
        # bytedata = cupy.unpackbits(cupy.asarray(bitdata), bitorder="little")
        bytedata = cupy.asarray(column.base_data)

        out = ak.contents.NumpyArray(
            cupy.asarray(bytedata).view(cupy.bool_),
            parameters=None,
            backend=CupyBackend.instance(),
        )
        return recurse_finalize(out, column, validbits, generate_bitmasks)

    elif isinstance(arrow_type, pyarrow.lib.DataType):
        validbits = column.base_mask
        dt = arrow_type.to_pandas_dtype()

        out = ak.contents.NumpyArray(
            cupy.asarray(column.base_data).view(dt),
            parameters=None,
            backend=CupyBackend.instance(),
        )
        return recurse_finalize(out, column, validbits, generate_bitmasks)

    else:
        raise TypeError(f"unrecognized Arrow array type: {arrow_type!r}")


def handle_cudf(cudf_series: cudf.core.series.Series, generate_bitmasks):
    column = cudf_series._data[cudf_series.name]
    dtype = column.dtype
    if isinstance(dtype, numpy.dtype):
        arrow_type = pyarrow.from_numpy_dtype(dtype)
    else:
        arrow_type = dtype.to_arrow()
    return recurse(column, arrow_type, generate_bitmasks)


def cudf_to_awkward(
    cudf_series: cudf.core.series.Series,
    generate_bitmasks=False,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    ctx = ak._layout.HighLevelContext(behavior=behavior, attrs=attrs).finalize()

    out = handle_cudf(cudf_series, generate_bitmasks)
    if isinstance(out, ak.contents.UnmaskedArray):
        out = remove_optiontype(out)

    def remove_revertable(layout, **kwargs):
        if hasattr(layout, "__pyarrow_original"):
            del layout.__pyarrow_original

    ak._do.recursively_apply(out, remove_revertable)

    return ctx.wrap(out, highlevel=highlevel)


######################### testing


if __name__ == "__main__":
    # tests numerics, lists, records, and option-type, but not union-type
    examples = [
        [False, True, True],  # booleans are special (1-bit)
        [1.1, 2.2, 3.3],
        [[False, True, True], [], [True, False]],
        [[1, 2, 3], [], [4, 5]],
        [[[1, 2], [3]], [], [[]], [[4], [], [5, 6, 7]], [[8, 9]]],
        [{"x": 1}, {"x": 2}, {"x": 3}],
        [{"x": 1.1, "y": []}, {"x": 2.2, "y": [1]}, {"x": 3.3, "y": [1, 2]}],
        [[{"x": 1}, {"x": 2}, {"x": 3}], [], [{"x": 4}, {"x": 5}]],
        [False, True, None, True],
        [1.1, 2.2, None, 3.3],
        [[False, True, None, True], [], [True, False]],
        [[False, True, True], None, [], [True, False]],
        [[1, 2, None, 3], [], [4, 5]],
        [[1, 2, 3], None, [], [4, 5]],
        [[[1, 2, None], [3]], [], [[]], [[4], [], [5, 6, 7]], [[8, 9]]],
        [[[1, 2], None, [3]], [], [[]], [[4], [], [5, 6, 7]], [[8, 9]]],
        [[[1, 2], [3]], None, [], [[]], [[4], [], [5, 6, 7]], [[8, 9]]],
        [{"x": 1}, {"x": None}, {"x": 3}],
        [{"x": 1}, {"x": 2}, None, {"x": 3}],
        [{"x": 1.1, "y": []}, {"x": None, "y": [1]}, {"x": 3.3, "y": [1, 2]}],
        [{"x": 1.1, "y": []}, {"x": 2.2, "y": [1, None]}, {"x": 3.3, "y": [1, 2]}],
        [{"x": 1.1, "y": []}, {"x": 2.2, "y": [1]}, None, {"x": 3.3, "y": [1, 2]}],
        [[{"x": 1}, {"x": None}, {"x": 3}], [], [{"x": 4}, {"x": 5}]],
        [[{"x": 1}, {"x": 2}, None, {"x": 3}], [], [{"x": 4}, {"x": 5}]],
        [[{"x": 1}, {"x": 2}, {"x": 3}], None, [], [{"x": 4}, {"x": 5}]],
    ]

    for example in examples:
        df = cudf.DataFrame({"column": example})

        pyarrow_array = df._data["column"].to_arrow()
        assert pyarrow_array.tolist() == example

        awkward_array = pyarrow_to_awkward(pyarrow_array)
        assert awkward_array.tolist() == example

    for example in examples:
        print(f"---- {example}")
        df = cudf.DataFrame({"column": example})

        awkward_array = cudf_to_awkward(df["column"])
        assert ak.backend(awkward_array) == "cuda"
        assert awkward_array.tolist() == example, awkward_array.show(type=True)
