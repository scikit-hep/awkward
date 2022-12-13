# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import json
from collections.abc import Iterable, Sized

import numpy

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()

try:
    import pyarrow

    error_message = None

except ModuleNotFoundError:
    pyarrow = None
    error_message = """to use {0}, you must install pyarrow:

    pip install pyarrow

or

    conda install -c conda-forge pyarrow
"""

else:
    if ak._util.parse_version(pyarrow.__version__) < ak._util.parse_version("7.0.0"):
        pyarrow = None
        error_message = "pyarrow 7.0.0 or later required for {0}"


def import_pyarrow(name):
    if pyarrow is None:
        raise ImportError(error_message.format(name))
    return pyarrow


def import_pyarrow_parquet(name):
    if pyarrow is None:
        raise ImportError(error_message.format(name))

    import pyarrow.parquet as out

    return out


def import_fsspec(name):
    try:
        import fsspec

    except ModuleNotFoundError as err:
        raise ImportError(
            f"""to use {name}, you must install fsspec:

    pip install fsspec

or

    conda install -c conda-forge fsspec
"""
        ) from err

    import_pyarrow_parquet(name)

    return fsspec


if pyarrow is not None:

    class AwkwardArrowArray(pyarrow.ExtensionArray):
        def to_pylist(self):
            out = super().to_pylist()
            if (
                isinstance(self.type, AwkwardArrowType)
                and self.type.node_type == "RecordArray"
                and self.type.record_is_tuple is True
            ):
                for i, x in enumerate(out):
                    if x is not None:
                        out[i] = tuple(x[str(j)] for j in range(len(x)))
            return out

    class AwkwardArrowType(pyarrow.ExtensionType):
        def __init__(
            self,
            storage_type,
            mask_type,
            node_type,
            mask_parameters,
            node_parameters,
            record_is_tuple,
            record_is_scalar,
        ):
            self._mask_type = mask_type
            self._node_type = node_type
            self._mask_parameters = mask_parameters
            self._node_parameters = node_parameters
            self._record_is_tuple = record_is_tuple
            self._record_is_scalar = record_is_scalar
            super().__init__(storage_type, "awkward")

        def __str__(self):
            return "ak:" + str(self.storage_type)

        def __repr__(self):
            return f"awkward<{self.storage_type!r}>"

        @property
        def mask_type(self):
            return self._mask_type

        @property
        def node_type(self):
            return self._node_type

        @property
        def mask_parameters(self):
            return self._mask_parameters

        @property
        def node_parameters(self):
            return self._node_parameters

        @property
        def record_is_tuple(self):
            return self._record_is_tuple

        @property
        def record_is_scalar(self):
            return self._record_is_scalar

        def __arrow_ext_class__(self):
            return AwkwardArrowArray

        def __arrow_ext_serialize__(self):
            return json.dumps(
                {
                    "mask_type": self._mask_type,
                    "node_type": self._node_type,
                    "mask_parameters": self._mask_parameters,
                    "node_parameters": self._node_parameters,
                    "record_is_tuple": self._record_is_tuple,
                    "record_is_scalar": self._record_is_scalar,
                }
            ).encode(errors="surrogatescape")

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            metadata = json.loads(serialized.decode(errors="surrogatescape"))
            return cls(
                storage_type,
                metadata["mask_type"],
                metadata["node_type"],
                metadata["mask_parameters"],
                metadata["node_parameters"],
                metadata["record_is_tuple"],
                metadata["record_is_scalar"],
            )

        @property
        def num_buffers(self):
            return self.storage_type.num_buffers

        @property
        def num_fields(self):
            return self.storage_type.num_fields

    pyarrow.register_extension_type(
        AwkwardArrowType(pyarrow.null(), None, None, None, None, None, None)
    )

    # order is important; _string_like[:2] vs _string_like[::2]
    _string_like = (
        pyarrow.string(),
        pyarrow.large_string(),
        pyarrow.binary(),
        pyarrow.large_binary(),
    )

    _pyarrow_to_numpy_dtype = {
        pyarrow.date32(): (True, np.dtype("M8[D]")),
        pyarrow.date64(): (False, np.dtype("M8[ms]")),
        pyarrow.time32("s"): (True, np.dtype("M8[s]")),
        pyarrow.time32("ms"): (True, np.dtype("M8[ms]")),
        pyarrow.time64("us"): (False, np.dtype("M8[us]")),
        pyarrow.time64("ns"): (False, np.dtype("M8[ns]")),
        pyarrow.timestamp("s"): (False, np.dtype("M8[s]")),
        pyarrow.timestamp("ms"): (False, np.dtype("M8[ms]")),
        pyarrow.timestamp("us"): (False, np.dtype("M8[us]")),
        pyarrow.timestamp("ns"): (False, np.dtype("M8[ns]")),
        pyarrow.duration("s"): (False, np.dtype("m8[s]")),
        pyarrow.duration("ms"): (False, np.dtype("m8[ms]")),
        pyarrow.duration("us"): (False, np.dtype("m8[us]")),
        pyarrow.duration("ns"): (False, np.dtype("m8[ns]")),
    }

if not ak._util.numpy_at_least("1.17.0"):

    def packbits(bytearray, lsb_order=True):
        if lsb_order:
            if len(bytearray) % 8 == 0:
                ready_to_pack = bytearray
            else:
                ready_to_pack = numpy.empty(
                    int(numpy.ceil(len(bytearray) / 8.0)) * 8,
                    dtype=bytearray.dtype,
                )
                ready_to_pack[: len(bytearray)] = bytearray
                ready_to_pack[len(bytearray) :] = 0
            return numpy.packbits(ready_to_pack.reshape(-1, 8)[:, ::-1].reshape(-1))
        else:
            return numpy.packbits(bytearray)

    def unpackbits(bitarray, lsb_order=True):
        ready_to_bitswap = numpy.unpackbits(bitarray)
        if lsb_order:
            return ready_to_bitswap.reshape(-1, 8)[:, ::-1].reshape(-1)
        else:
            return ready_to_bitswap

else:

    def packbits(bytearray, lsb_order=True):
        return numpy.packbits(bytearray, bitorder=("little" if lsb_order else "big"))

    def unpackbits(bitarray, lsb_order=True):
        return numpy.unpackbits(bitarray, bitorder=("little" if lsb_order else "big"))


def and_validbytes(validbytes1, validbytes2):
    if validbytes1 is None:
        return validbytes2
    elif validbytes2 is None:
        return validbytes1
    else:
        return validbytes1 & validbytes2


def to_validbits(validbytes):
    if validbytes is None:
        return None
    else:
        return pyarrow.py_buffer(packbits(validbytes))


def to_length(nparray, length):
    if len(nparray) < length:
        out = numpy.empty(length, dtype=nparray.dtype)
        out[: len(nparray)] = nparray
    else:
        out = nparray
    return pyarrow.py_buffer(out)


def to_null_count(validbytes, count_nulls):
    if validbytes is None or not count_nulls:
        return -1
    else:
        return len(validbytes) - numpy.count_nonzero(validbytes)


def to_awkwardarrow_storage_types(arrowtype):
    if isinstance(arrowtype, AwkwardArrowType):
        return arrowtype, arrowtype.storage_type
    else:
        return None, arrowtype


def node_parameters(awkwardarrow_type):
    if isinstance(awkwardarrow_type, AwkwardArrowType):
        return awkwardarrow_type.node_parameters
    else:
        return None


def mask_parameters(awkwardarrow_type):
    if isinstance(awkwardarrow_type, AwkwardArrowType):
        return awkwardarrow_type.mask_parameters
    else:
        return None


def revertable(modified, original):
    modified.__pyarrow_original = original
    return modified


def popbuffers_finalize(
    out, array, validbits, awkwardarrow_type, generate_bitmasks, fix_offsets=True
):
    # Every buffer from Arrow must be offsets-corrected.
    if fix_offsets and (array.offset != 0 or len(array) != len(out)):
        out = out[array.offset : array.offset + len(array)]

    # Everything must leave popbuffers as option-type; the mask_node will be
    # removed by the next level up in popbuffers recursion if appropriate.

    if isinstance(awkwardarrow_type, AwkwardArrowType):
        if awkwardarrow_type.mask_type == "UnmaskedArray":
            assert validbits is None or numpy.all(
                numpy.frombuffer(validbits, np.uint8)[: len(out) // 8] == 0xFF
            )
            return revertable(
                ak.contents.UnmaskedArray.simplified(
                    out, parameters=awkwardarrow_type.mask_parameters
                ),
                out,
            )

        else:
            if validbits is None:
                if generate_bitmasks:
                    return revertable(
                        ak.contents.BitMaskedArray.simplified(
                            # ceildiv(len(out), 8) = -(len(out) // -8)
                            ak.index.IndexU8(
                                numpy.full(-(len(out) // -8), np.uint8(0xFF))
                            ),
                            out,
                            valid_when=True,
                            length=len(out),
                            lsb_order=True,
                            parameters=awkwardarrow_type.mask_parameters,
                        ),
                        out,
                    )
                else:
                    return revertable(
                        ak.contents.UnmaskedArray.simplified(
                            out, parameters=awkwardarrow_type.mask_parameters
                        ),
                        out,
                    )
            else:
                return revertable(
                    ak.contents.BitMaskedArray.simplified(
                        ak.index.IndexU8(numpy.frombuffer(validbits, dtype=np.uint8)),
                        out,
                        valid_when=True,
                        length=len(out),
                        lsb_order=True,
                        parameters=awkwardarrow_type.mask_parameters,
                    ),
                    out,
                )

    else:
        if validbits is None and generate_bitmasks:
            # ceildiv(len(out), 8) = -(len(out) // -8)
            validbits = numpy.full(-(len(out) // -8), np.uint8(0xFF))

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


def form_popbuffers_finalize(out, awkwardarrow_type):
    if (
        isinstance(awkwardarrow_type, AwkwardArrowType)
        and awkwardarrow_type.mask_type == "UnmaskedArray"
    ):
        return revertable(
            ak.forms.UnmaskedForm.simplified(
                out, parameters=awkwardarrow_type.mask_parameters
            ),
            out,
        )

    else:
        return revertable(
            ak.forms.BitMaskedForm.simplified(
                "u8",
                out,
                valid_when=True,
                lsb_order=True,
                parameters=mask_parameters(awkwardarrow_type),
            ),
            out,
        )


def popbuffers(paarray, awkwardarrow_type, storage_type, buffers, generate_bitmasks):
    # Start by removing the ExtensionArray wrapper.
    if awkwardarrow_type is not None:
        paarray = paarray.storage

    ### Beginning of the big if-elif-elif chain!

    if isinstance(storage_type, pyarrow.lib.PyExtensionType):
        raise ak._errors.wrap_error(
            ValueError(
                "Arrow arrays containing pickled Python objects can't be converted into Awkward Arrays"
            )
        )

    elif isinstance(storage_type, pyarrow.lib.ExtensionType):
        # AwkwardArrowType should already be unwrapped; this must be some other ExtensionType.
        assert not isinstance(storage_type, AwkwardArrowType)
        # In that case, just ignore its logical type and use its storage type.
        return popbuffers(
            paarray,
            awkwardarrow_type,
            storage_type.storage_type,
            buffers,
            generate_bitmasks,
        )

    elif isinstance(storage_type, pyarrow.lib.DictionaryType):
        masked_index = popbuffers(
            paarray.indices,
            None,
            storage_type.index_type,
            buffers,
            generate_bitmasks,
        )
        index = masked_index.content.data

        if not isinstance(masked_index, ak.contents.UnmaskedArray):
            mask = masked_index.mask_as_bool(valid_when=False)
            if mask.any():
                index = numpy.array(index, copy=True)
                index[mask] = -1

        content = handle_arrow(paarray.dictionary, generate_bitmasks)

        parameters = ak._util.merge_parameters(
            mask_parameters(awkwardarrow_type), node_parameters(awkwardarrow_type)
        )
        if parameters is None:
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

        a, b = to_awkwardarrow_storage_types(storage_type.value_type)
        akcontent = popbuffers(paarray.values, a, b, buffers, generate_bitmasks)

        if not storage_type.value_field.nullable:
            # strip the dummy option-type node
            akcontent = remove_optiontype(akcontent)

        out = ak.contents.RegularArray(
            akcontent,
            storage_type.list_size,
            parameters=node_parameters(awkwardarrow_type),
        )
        return popbuffers_finalize(
            out, paarray, validbits, awkwardarrow_type, generate_bitmasks
        )

    elif isinstance(storage_type, (pyarrow.lib.LargeListType, pyarrow.lib.ListType)):
        assert storage_type.num_buffers == 2
        validbits = buffers.pop(0)
        paoffsets = buffers.pop(0)

        if isinstance(storage_type, pyarrow.lib.LargeListType):
            akoffsets = ak.index.Index64(numpy.frombuffer(paoffsets, dtype=np.int64))
        else:
            akoffsets = ak.index.Index32(numpy.frombuffer(paoffsets, dtype=np.int32))

        a, b = to_awkwardarrow_storage_types(storage_type.value_type)
        akcontent = popbuffers(paarray.values, a, b, buffers, generate_bitmasks)

        if not storage_type.value_field.nullable:
            # strip the dummy option-type node
            akcontent = remove_optiontype(akcontent)

        out = ak.contents.ListOffsetArray(
            akoffsets, akcontent, parameters=node_parameters(awkwardarrow_type)
        )
        return popbuffers_finalize(
            out, paarray, validbits, awkwardarrow_type, generate_bitmasks
        )

    elif isinstance(storage_type, pyarrow.lib.MapType):
        # FIXME: make a ListOffsetArray of 2-tuples with __array__ == "sorted_map".
        # (Make sure the keys are sorted).
        raise ak._errors.wrap_error(NotImplementedError)

    elif isinstance(
        storage_type, (pyarrow.lib.Decimal128Type, pyarrow.lib.Decimal256Type)
    ):
        # Note: Decimal128Type and Decimal256Type are subtypes of FixedSizeBinaryType.
        # NumPy doesn't support decimal: https://github.com/numpy/numpy/issues/9789
        raise ak._errors.wrap_error(
            ValueError(
                "Arrow arrays containing pyarrow.decimal128 or pyarrow.decimal256 types can't be converted into Awkward Arrays"
            )
        )

    elif isinstance(storage_type, pyarrow.lib.FixedSizeBinaryType):
        assert storage_type.num_buffers == 2
        validbits = buffers.pop(0)
        pacontent = buffers.pop(0)

        parameters = node_parameters(awkwardarrow_type)
        if parameters is None:
            parameters = {"__array__": "bytestring"}
        sub_parameters = {"__array__": "byte"}

        out = ak.contents.RegularArray(
            ak.contents.NumpyArray(
                numpy.frombuffer(pacontent, dtype=np.uint8),
                parameters=sub_parameters,
                backend=ak._backends.NumpyBackend.instance(),
            ),
            storage_type.byte_width,
            parameters=parameters,
        )
        return popbuffers_finalize(
            out, paarray, validbits, awkwardarrow_type, generate_bitmasks
        )

    elif storage_type in _string_like:
        assert storage_type.num_buffers == 3
        validbits = buffers.pop(0)
        paoffsets = buffers.pop(0)
        pacontent = buffers.pop(0)

        if storage_type in _string_like[::2]:
            akoffsets = ak.index.Index32(numpy.frombuffer(paoffsets, dtype=np.int32))
        else:
            akoffsets = ak.index.Index64(numpy.frombuffer(paoffsets, dtype=np.int64))

        parameters = node_parameters(awkwardarrow_type)

        if storage_type in _string_like[:2]:
            if parameters is None:
                parameters = {"__array__": "string"}
            sub_parameters = {"__array__": "char"}
        else:
            if parameters is None:
                parameters = {"__array__": "bytestring"}
            sub_parameters = {"__array__": "byte"}

        out = ak.contents.ListOffsetArray(
            akoffsets,
            ak.contents.NumpyArray(
                numpy.frombuffer(pacontent, dtype=np.uint8),
                parameters=sub_parameters,
                backend=ak._backends.NumpyBackend.instance(),
            ),
            parameters=parameters,
        )
        return popbuffers_finalize(
            out, paarray, validbits, awkwardarrow_type, generate_bitmasks
        )

    elif isinstance(storage_type, pyarrow.lib.StructType):
        assert storage_type.num_buffers == 1
        validbits = buffers.pop(0)

        keys = []
        contents = []
        for i in range(storage_type.num_fields):
            field = storage_type[i]
            field_name = field.name
            keys.append(field_name)

            a, b = to_awkwardarrow_storage_types(field.type)
            akcontent = popbuffers(
                paarray.field(field_name), a, b, buffers, generate_bitmasks
            )
            if not field.nullable:
                # strip the dummy option-type node
                akcontent = remove_optiontype(akcontent)
            contents.append(akcontent)

        if awkwardarrow_type is not None and awkwardarrow_type.record_is_tuple:
            keys = None

        out = ak.contents.RecordArray(
            contents,
            keys,
            length=len(paarray),
            parameters=node_parameters(awkwardarrow_type),
        )
        return popbuffers_finalize(
            out,
            paarray,
            validbits,
            awkwardarrow_type,
            generate_bitmasks,
            fix_offsets=False,
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
            a, b = to_awkwardarrow_storage_types(field.type)
            akcontent = popbuffers(paarray.field(i), a, b, buffers, generate_bitmasks)

            if not field.nullable:
                # strip the dummy option-type node
                akcontent = remove_optiontype(akcontent)
            akcontents.append(akcontent)

        out = ak.contents.UnionArray.simplified(
            ak.index.Index8(nptags),
            ak.index.Index32(npindex),
            akcontents,
            parameters=node_parameters(awkwardarrow_type),
        )
        return popbuffers_finalize(
            out, paarray, None, awkwardarrow_type, generate_bitmasks
        )

    elif storage_type == pyarrow.null():
        validbits = buffers.pop(0)
        assert storage_type.num_fields == 0

        # This is already an option-type and offsets-corrected, so no popbuffers_finalize.
        return ak.contents.IndexedOptionArray(
            ak.index.Index64(numpy.full(len(paarray), -1, dtype=np.int64)),
            ak.contents.EmptyArray(parameters=node_parameters(awkwardarrow_type)),
            parameters=mask_parameters(awkwardarrow_type),
        )

    elif storage_type == pyarrow.bool_():
        assert storage_type.num_buffers == 2
        validbits = buffers.pop(0)
        bitdata = buffers.pop(0)

        bytedata = unpackbits(numpy.frombuffer(bitdata, dtype=np.uint8))

        out = ak.contents.NumpyArray(
            bytedata.view(np.bool_),
            parameters=node_parameters(awkwardarrow_type),
            backend=ak._backends.NumpyBackend.instance(),
        )
        return popbuffers_finalize(
            out, paarray, validbits, awkwardarrow_type, generate_bitmasks
        )

    elif isinstance(storage_type, pyarrow.lib.DataType):
        assert storage_type.num_buffers == 2
        validbits = buffers.pop(0)
        data = buffers.pop(0)

        to64, dt = _pyarrow_to_numpy_dtype.get(str(storage_type), (False, None))
        if to64:
            data = numpy.frombuffer(data, dtype=np.int32).astype(np.int64)
        if dt is None:
            dt = storage_type.to_pandas_dtype()

        out = ak.contents.NumpyArray(
            numpy.frombuffer(data, dtype=dt),
            parameters=node_parameters(awkwardarrow_type),
            backend=ak._backends.NumpyBackend.instance(),
        )
        return popbuffers_finalize(
            out, paarray, validbits, awkwardarrow_type, generate_bitmasks
        )

    else:
        raise ak._errors.wrap_error(
            TypeError(f"unrecognized Arrow array type: {storage_type!r}")
        )


def form_popbuffers(awkwardarrow_type, storage_type):
    ### Beginning of the big if-elif-elif chain!

    if isinstance(storage_type, pyarrow.lib.PyExtensionType):
        raise ak._errors.wrap_error(
            ValueError(
                "Arrow arrays containing pickled Python objects can't be converted into Awkward Arrays"
            )
        )

    elif isinstance(storage_type, pyarrow.lib.ExtensionType):
        # AwkwardArrowType should already be unwrapped; this must be some other ExtensionType.
        assert not isinstance(storage_type, AwkwardArrowType)
        # In that case, just ignore its logical type and use its storage type.
        return form_popbuffers(awkwardarrow_type, storage_type.storage_type)

    elif isinstance(storage_type, pyarrow.lib.DictionaryType):
        index_type = storage_type.index_type.to_pandas_dtype()
        if index_type is np.int64:
            index = "i64"
        elif index_type is np.uint32:
            index = "u32"
        elif index_type is np.int32:
            index = "i32"
        else:
            raise ak._errors.wrap_error(
                TypeError(f"unrecognized Arrow DictionaryType index type: {index_type}")
            )

        a, b = to_awkwardarrow_storage_types(storage_type.value_type)
        content = form_popbuffers(a, b)

        parameters = ak._util.merge_parameters(
            mask_parameters(awkwardarrow_type), node_parameters(awkwardarrow_type)
        )
        if parameters is None:
            parameters = {"__array__": "categorical"}

        return revertable(
            ak.forms.IndexedOptionForm.simplified(
                index,
                content,
                parameters=parameters,
            ),
            ak.forms.IndexedForm(
                index,
                form_remove_optiontype(content) if content.is_option else content,
                parameters=parameters,
            ),
        )

    elif isinstance(storage_type, pyarrow.lib.FixedSizeListType):
        a, b = to_awkwardarrow_storage_types(storage_type.value_type)
        akcontent = form_popbuffers(a, b)

        if not storage_type.value_field.nullable:
            # strip the dummy option-type node
            akcontent = form_remove_optiontype(akcontent)

        out = ak.forms.RegularForm(
            akcontent,
            storage_type.list_size,
            parameters=node_parameters(awkwardarrow_type),
        )
        return form_popbuffers_finalize(out, awkwardarrow_type)

    elif isinstance(storage_type, (pyarrow.lib.LargeListType, pyarrow.lib.ListType)):
        if isinstance(storage_type, pyarrow.lib.LargeListType):
            akoffsets = "i64"
        else:
            akoffsets = "i32"

        a, b = to_awkwardarrow_storage_types(storage_type.value_type)
        akcontent = form_popbuffers(a, b)

        if not storage_type.value_field.nullable:
            # strip the dummy option-type node
            akcontent = form_remove_optiontype(akcontent)

        out = ak.forms.ListOffsetForm(
            akoffsets, akcontent, parameters=node_parameters(awkwardarrow_type)
        )
        return form_popbuffers_finalize(out, awkwardarrow_type)

    elif isinstance(storage_type, pyarrow.lib.MapType):
        raise ak._errors.wrap_error(NotImplementedError)

    elif isinstance(
        storage_type, (pyarrow.lib.Decimal128Type, pyarrow.lib.Decimal256Type)
    ):
        # Note: Decimal128Type and Decimal256Type are subtypes of FixedSizeBinaryType.
        # NumPy doesn't support decimal: https://github.com/numpy/numpy/issues/9789
        raise ak._errors.wrap_error(
            ValueError(
                "Arrow arrays containing pyarrow.decimal128 or pyarrow.decimal256 types can't be converted into Awkward Arrays"
            )
        )

    elif isinstance(storage_type, pyarrow.lib.FixedSizeBinaryType):
        parameters = node_parameters(awkwardarrow_type)
        if parameters is None:
            parameters = {"__array__": "bytestring"}
        sub_parameters = {"__array__": "byte"}

        out = ak.forms.RegularForm(
            ak.forms.NumpyForm("uint8", parameters=sub_parameters),
            storage_type.byte_width,
            parameters=parameters,
        )
        return form_popbuffers_finalize(out, awkwardarrow_type)

    elif storage_type in _string_like:
        if storage_type in _string_like[::2]:
            akoffsets = "i32"
        else:
            akoffsets = "i64"

        parameters = node_parameters(awkwardarrow_type)

        if storage_type in _string_like[:2]:
            if parameters is None:
                parameters = {"__array__": "string"}
            sub_parameters = {"__array__": "char"}
        else:
            if parameters is None:
                parameters = {"__array__": "bytestring"}
            sub_parameters = {"__array__": "byte"}

        out = ak.forms.ListOffsetForm(
            akoffsets,
            ak.forms.NumpyForm("uint8", parameters=sub_parameters),
            parameters=parameters,
        )
        return form_popbuffers_finalize(out, awkwardarrow_type)

    elif isinstance(storage_type, pyarrow.lib.StructType):
        keys = []
        contents = []
        for i in range(storage_type.num_fields):
            field = storage_type[i]
            field_name = field.name
            keys.append(field_name)

            a, b = to_awkwardarrow_storage_types(field.type)
            akcontent = form_popbuffers(a, b)
            if not field.nullable:
                # strip the dummy option-type node
                akcontent = form_remove_optiontype(akcontent)
            contents.append(akcontent)

        if awkwardarrow_type is not None and awkwardarrow_type.record_is_tuple:
            keys = None

        out = ak.forms.RecordForm(
            contents,
            keys,
            parameters=node_parameters(awkwardarrow_type),
        )
        return form_popbuffers_finalize(out, awkwardarrow_type)

    elif isinstance(storage_type, pyarrow.lib.UnionType):
        akcontents = []
        for i in range(storage_type.num_fields):
            field = storage_type[i]
            a, b = to_awkwardarrow_storage_types(field.type)
            akcontent = form_popbuffers(a, b)

            if not field.nullable:
                # strip the dummy option-type node
                akcontent = form_remove_optiontype(akcontent)
            akcontents.append(akcontent)

        out = ak.forms.UnionForm.simplified(
            "i8", "i32", akcontents, parameters=node_parameters(awkwardarrow_type)
        )
        return form_popbuffers_finalize(out, awkwardarrow_type)

    elif storage_type == pyarrow.null():
        # This is already an option-type, so no form_popbuffers_finalize.
        return ak.forms.IndexedOptionForm(
            "i64",
            ak.forms.EmptyForm(parameters=node_parameters(awkwardarrow_type)),
            parameters=mask_parameters(awkwardarrow_type),
        )

    elif isinstance(storage_type, pyarrow.lib.DataType):
        _, dt = _pyarrow_to_numpy_dtype.get(str(storage_type), (False, None))
        if dt is None:
            dt = np.dtype(storage_type.to_pandas_dtype())

        out = ak.forms.NumpyForm(
            ak.types.numpytype.dtype_to_primitive(dt),
            parameters=node_parameters(awkwardarrow_type),
        )

        return form_popbuffers_finalize(out, awkwardarrow_type)

    else:
        raise ak._errors.wrap_error(
            TypeError(f"unrecognized Arrow array type: {storage_type!r}")
        )


def to_awkwardarrow_type(
    storage_type, use_extensionarray, record_is_scalar, mask, node
):
    if use_extensionarray:
        return AwkwardArrowType(
            storage_type,
            ak._util.direct_Content_subclass_name(mask),
            ak._util.direct_Content_subclass_name(node),
            None if mask is None else mask.parameters,
            None if node is None else node.parameters,
            node.is_tuple if isinstance(node, ak.contents.RecordArray) else None,
            record_is_scalar,
        )
    else:
        return storage_type


def remove_optiontype(akarray):
    return akarray.__pyarrow_original


def form_remove_optiontype(akform):
    return akform.__pyarrow_original


def handle_arrow(obj, generate_bitmasks=False, pass_empty_field=False):
    if isinstance(obj, pyarrow.lib.Array):
        buffers = obj.buffers()

        awkwardarrow_type, storage_type = to_awkwardarrow_storage_types(obj.type)

        out = popbuffers(
            obj, awkwardarrow_type, storage_type, buffers, generate_bitmasks
        )
        assert len(buffers) == 0
        return out

    elif isinstance(obj, pyarrow.lib.ChunkedArray):
        layouts = [handle_arrow(x, generate_bitmasks) for x in obj.chunks if len(x) > 0]

        if len(layouts) == 1:
            return layouts[0]
        else:
            return ak.operations.concatenate(layouts, highlevel=False)

    elif isinstance(obj, pyarrow.lib.RecordBatch):
        if pass_empty_field and list(obj.schema.names) == [""]:
            layout = handle_arrow(obj.column(0), generate_bitmasks)
            if not obj.schema.field(0).nullable:
                return remove_optiontype(layout)
            else:
                return layout
        else:
            record_is_optiontype = False
            optiontype_fields = []
            record_is_scalar = False
            optiontype_parameters = None
            recordtype_parameters = None
            if (
                obj.schema.metadata is not None
                and b"ak:parameters" in obj.schema.metadata
            ):
                for x in json.loads(obj.schema.metadata[b"ak:parameters"]):
                    (key,) = x.keys()
                    (value,) = x.values()
                    if key == "optiontype_fields":
                        optiontype_fields = value
                    elif key == "record_is_scalar":
                        record_is_scalar = value
                    elif key in (
                        "UnmaskedArray",
                        "BitMaskedArray",
                        "ByteMaskedArray",
                        "IndexedOptionArray",
                    ):
                        record_is_optiontype = True
                        optiontype_parameters = value
                    elif key == "RecordArray":
                        recordtype_parameters = value

            record_mask = None
            contents = []
            for i in range(obj.num_columns):
                field = obj.schema.field(i)
                layout = handle_arrow(obj.column(i), generate_bitmasks)
                if record_is_optiontype:
                    if record_mask is None:
                        record_mask = layout.mask_as_bool(valid_when=False)
                    else:
                        record_mask &= layout.mask_as_bool(valid_when=False)
                if (
                    record_is_optiontype and field.name not in optiontype_fields
                ) or not field.nullable:
                    contents.append(remove_optiontype(layout))
                else:
                    contents.append(layout)

            out = ak.contents.RecordArray(
                contents,
                obj.schema.names,
                length=len(obj),
                parameters=recordtype_parameters,
            )

            if record_is_scalar:
                return out._getitem_at(0)

            if record_is_optiontype and record_mask is None and generate_bitmasks:
                record_mask = numpy.zeros(len(out), dtype=np.bool_)

            if record_is_optiontype and record_mask is None:
                return ak.contents.UnmaskedArray.simplified(
                    out, parameters=optiontype_parameters
                )

            elif record_is_optiontype:
                return ak.contents.ByteMaskedArray.simplified(
                    ak.index.Index8(record_mask),
                    out,
                    valid_when=False,
                    parameters=optiontype_parameters,
                )

            else:
                return out

    elif isinstance(obj, pyarrow.lib.Table):
        batches = obj.combine_chunks().to_batches()
        if len(batches) == 0:
            # FIXME: create a zero-length array with the right type
            raise ak._errors.wrap_error(NotImplementedError)
        elif len(batches) == 1:
            return handle_arrow(batches[0], generate_bitmasks, pass_empty_field)
        else:
            arrays = [
                handle_arrow(batch, generate_bitmasks, pass_empty_field)
                for batch in batches
                if len(batch) > 0
            ]
            return ak.operations.concatenate(arrays, highlevel=False)

    elif (
        isinstance(obj, Iterable)
        and isinstance(obj, Sized)
        and len(obj) > 0
        and all(isinstance(x, pyarrow.lib.RecordBatch) for x in obj)
        and any(len(x) > 0 for x in obj)
    ):
        chunks = []
        for batch in obj:
            chunk = handle_arrow(batch, generate_bitmasks, pass_empty_field)
            if len(chunk) > 0:
                chunks.append(chunk)
        if len(chunks) == 1:
            return chunks[0]
        else:
            return ak.operations.concatenate(chunks, highlevel=False)

    elif isinstance(obj, Iterable) and len(obj) == 0:
        return ak.contents.RecordArray([], [], length=0)

    else:
        raise ak._errors.wrap_error(TypeError(f"unrecognized Arrow type: {type(obj)}"))


def form_handle_arrow(schema, pass_empty_field=False):
    if pass_empty_field and list(schema.names) == [""]:
        awkwardarrow_type, storage_type = to_awkwardarrow_storage_types(schema.types[0])
        akform = form_popbuffers(awkwardarrow_type, storage_type)
        if not schema.field(0).nullable:
            return form_remove_optiontype(akform)
        else:
            return akform

    else:
        record_is_optiontype = False
        optiontype_fields = []
        optiontype_parameters = None
        recordtype_parameters = None
        if schema.metadata is not None and b"ak:parameters" in schema.metadata:
            for x in json.loads(schema.metadata[b"ak:parameters"]):
                (key,) = x.keys()
                (value,) = x.values()
                if key == "optiontype_fields":
                    optiontype_fields = value
                elif key in (
                    "UnmaskedArray",
                    "BitMaskedArray",
                    "ByteMaskedArray",
                    "IndexedOptionArray",
                ):
                    record_is_optiontype = True
                    optiontype_parameters = value
                elif key == "RecordArray":
                    recordtype_parameters = value

        forms = []
        for i, arrowtype in enumerate(schema.types):
            field = schema.field(i)
            awkwardarrow_type, storage_type = to_awkwardarrow_storage_types(arrowtype)
            akform = form_popbuffers(awkwardarrow_type, storage_type)

            if (
                record_is_optiontype and field.name not in optiontype_fields
            ) or not field.nullable:
                forms.append(form_remove_optiontype(akform))
            else:
                forms.append(akform)

        out = ak.forms.RecordForm(
            forms, list(schema.names), parameters=recordtype_parameters
        )

        if record_is_optiontype:
            return ak.forms.ByteMaskedForm.simplified(
                "i8", out, valid_when=False, parameters=optiontype_parameters
            )

        else:
            return out
