import cudf
import cupy as cp
import pyarrow as pa
import awkward as ak


######################### stripped-down copy of src/awkward/_connect/pyarrow.py


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
        validbits = numpy.full(-(len(out) // -8), np.uint8(0xFF), dtype=np.uint8)

    if validbits is None:
        return revertable(ak.contents.UnmaskedArray.simplified(out), out)
    else:
        return revertable(
            ak.contents.BitMaskedArray.simplified(
                ak.index.IndexU8(numpy.frombuffer(validbits, dtype=np.uint8)),
                out,
                valid_when=True,
                length=len(out),
                lsb_order=True,
            ),
            out,
        )


def popbuffers(paarray, storage_type, buffers, generate_bitmasks):
    ### Beginning of the big if-elif-elif chain!

    if isinstance(storage_type, pyarrow.lib.DictionaryType):
        masked_index = popbuffers(
            paarray.indices,
            storage_type.index_type,
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

    elif isinstance(storage_type, pyarrow.lib.FixedSizeListType):
        assert storage_type.num_buffers == 1
        validbits = buffers.pop(0)

        akcontent = popbuffers(
            paarray.values, storage_type.value_type, buffers, generate_bitmasks
        )

        if not storage_type.value_field.nullable:
            # strip the dummy option-type node
            akcontent = remove_optiontype(akcontent)

        out = ak.contents.RegularArray(
            akcontent,
            storage_type.list_size,
            parameters=None,
        )
        return popbuffers_finalize(out, paarray, validbits, generate_bitmasks)

    elif isinstance(storage_type, (pyarrow.lib.LargeListType, pyarrow.lib.ListType)):
        assert storage_type.num_buffers == 2
        validbits = buffers.pop(0)
        paoffsets = buffers.pop(0)

        if isinstance(storage_type, pyarrow.lib.LargeListType):
            akoffsets = ak.index.Index64(numpy.frombuffer(paoffsets, dtype=np.int64))
        else:
            akoffsets = ak.index.Index32(numpy.frombuffer(paoffsets, dtype=np.int32))

        akcontent = popbuffers(
            paarray.values, storage_type.value_type, buffers, generate_bitmasks
        )

        if not storage_type.value_field.nullable:
            # strip the dummy option-type node
            akcontent = remove_optiontype(akcontent)

        out = ak.contents.ListOffsetArray(akoffsets, akcontent, parameters=None)
        return popbuffers_finalize(out, paarray, validbits, generate_bitmasks)

    elif isinstance(storage_type, pyarrow.lib.MapType):
        # FIXME: make a ListOffsetArray of 2-tuples with __array__ == "sorted_map".
        # (Make sure the keys are sorted).
        raise NotImplementedError

    elif isinstance(
        storage_type, (pyarrow.lib.Decimal128Type, pyarrow.lib.Decimal256Type)
    ):
        # Note: Decimal128Type and Decimal256Type are subtypes of FixedSizeBinaryType.
        # NumPy doesn't support decimal: https://github.com/numpy/numpy/issues/9789
        raise ValueError(
            "Arrow arrays containing pyarrow.decimal128 or pyarrow.decimal256 types can't be converted into Awkward Arrays"
        )

    elif isinstance(storage_type, pyarrow.lib.FixedSizeBinaryType):
        assert storage_type.num_buffers == 2
        validbits = buffers.pop(0)
        pacontent = buffers.pop(0)

        parameters = {"__array__": "bytestring"}
        sub_parameters = {"__array__": "byte"}

        out = ak.contents.RegularArray(
            ak.contents.NumpyArray(
                numpy.frombuffer(pacontent, dtype=np.uint8),
                parameters=sub_parameters,
                backend=NumpyBackend.instance(),
            ),
            storage_type.byte_width,
            parameters=parameters,
        )
        return popbuffers_finalize(out, paarray, validbits, generate_bitmasks)

    elif storage_type in _string_like:
        assert storage_type.num_buffers == 3
        validbits = buffers.pop(0)
        paoffsets = buffers.pop(0)
        pacontent = buffers.pop(0)

        if storage_type in _string_like[::2]:
            akoffsets = ak.index.Index32(numpy.frombuffer(paoffsets, dtype=np.int32))
        else:
            akoffsets = ak.index.Index64(numpy.frombuffer(paoffsets, dtype=np.int64))

        if storage_type in _string_like[:2]:
            parameters = {"__array__": "string"}
            sub_parameters = {"__array__": "char"}
        else:
            parameters = {"__array__": "bytestring"}
            sub_parameters = {"__array__": "byte"}

        out = ak.contents.ListOffsetArray(
            akoffsets,
            ak.contents.NumpyArray(
                numpy.frombuffer(pacontent, dtype=np.uint8),
                parameters=sub_parameters,
                backend=NumpyBackend.instance(),
            ),
            parameters=parameters,
        )
        return popbuffers_finalize(out, paarray, validbits, generate_bitmasks)

    elif isinstance(storage_type, pyarrow.lib.StructType):
        assert storage_type.num_buffers == 1
        validbits = buffers.pop(0)

        keys = []
        contents = []
        for i in range(storage_type.num_fields):
            field = storage_type[i]
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

    elif isinstance(storage_type, pyarrow.lib.UnionType):
        if isinstance(storage_type, pyarrow.lib.SparseUnionType):
            assert storage_type.num_buffers == 2
            validbits = buffers.pop(0)
            nptags = numpy.frombuffer(buffers.pop(0), dtype=np.int8)
            npindex = numpy.arange(len(nptags), dtype=np.int32)
        else:
            assert storage_type.num_buffers == 3
            validbits = buffers.pop(0)
            nptags = numpy.frombuffer(buffers.pop(0), dtype=np.int8)
            npindex = numpy.frombuffer(buffers.pop(0), dtype=np.int32)

        akcontents = []
        for i in range(storage_type.num_fields):
            field = storage_type[i]
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

    elif storage_type == pyarrow.null():
        validbits = buffers.pop(0)
        assert storage_type.num_fields == 0

        # This is already an option-type and offsets-corrected, so no popbuffers_finalize.
        return ak.contents.IndexedOptionArray(
            ak.index.Index64(numpy.full(len(paarray), -1, dtype=np.int64)),
            ak.contents.EmptyArray(parameters=None),
            parameters=None,
        )

    elif storage_type == pyarrow.bool_():
        assert storage_type.num_buffers == 2
        validbits = buffers.pop(0)
        bitdata = buffers.pop(0)

        bytedata = numpy.unpackbits(
            numpy.frombuffer(bitdata, dtype=np.uint8), bitorder="little"
        )

        out = ak.contents.NumpyArray(
            bytedata.view(np.bool_),
            parameters=None,
            backend=NumpyBackend.instance(),
        )
        return popbuffers_finalize(out, paarray, validbits, generate_bitmasks)

    elif isinstance(storage_type, pyarrow.lib.DataType):
        assert storage_type.num_buffers == 2
        validbits = buffers.pop(0)
        data = buffers.pop(0)

        to64, dt = _pyarrow_to_numpy_dtype.get(str(storage_type), (False, None))
        if to64:
            data = numpy.astype(numpy.frombuffer(data, dtype=np.int32), dtype=np.int64)
        if dt is None:
            dt = storage_type.to_pandas_dtype()

        out = ak.contents.NumpyArray(
            numpy.frombuffer(data, dtype=dt),
            parameters=None,
            backend=NumpyBackend.instance(),
        )
        return popbuffers_finalize(out, paarray, validbits, generate_bitmasks)

    else:
        raise TypeError(f"unrecognized Arrow array type: {storage_type!r}")


def handle_arrow(obj, generate_bitmasks, pass_empty_field):
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

    out = handle_arrow(pyarrow_array, generate_bitmasks, True)
    if isinstance(out, ak.contents.UnmaskedArray):
        out = remove_optiontype(out)

    def remove_revertable(layout, **kwargs):
        if hasattr(layout, "__pyarrow_original"):
            del layout.__pyarrow_original

    ak._do.recursively_apply(out, remove_revertable)

    return ctx.wrap(out, highlevel=highlevel)


if __name__ == "__main__":
    df = cudf.DataFrame({"record": [{"inner": [[3], [1, 2]], "simple": [8, None]}] * 6})


