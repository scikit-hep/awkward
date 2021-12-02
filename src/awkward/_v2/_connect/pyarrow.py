# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import distutils
import json

try:
    from collections.abc import Iterable, Sized
except ImportError:
    from collections import Iterable, Sized

import numpy

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()

try:
    import pyarrow

    error_message = None

except ImportError:
    pyarrow = None
    error_message = """to use {0}, you must install pyarrow:

    pip install pyarrow

or

    conda install -c conda-forge pyarrow
"""

else:
    if distutils.version.LooseVersion(
        pyarrow.__version__
    ) < distutils.version.LooseVersion("6.0.0"):
        pyarrow = None
        error_message = "pyarrow 6.0.0 or later required for {0}"


def import_pyarrow(name):
    if pyarrow is None:
        raise ImportError(error_message.format(name))
    return pyarrow


def import_pyarrow_parquet(name):
    if pyarrow is None:
        raise ImportError(error_message.format(name))

    import pyarrow.parquet

    return pyarrow.parquet


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
        ):
            self._mask_type = mask_type
            self._node_type = node_type
            self._mask_parameters = mask_parameters
            self._node_parameters = node_parameters
            self._record_is_tuple = record_is_tuple
            super(AwkwardArrowType, self).__init__(storage_type, "awkward")

        def __str__(self):
            return "ak:" + str(self.storage_type)

        def __repr__(self):
            return "awkward<{0}>".format(repr(self.storage_type))

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
            )

        @property
        def num_buffers(self):
            return self.storage_type.num_buffers

        @property
        def num_fields(self):
            return self.storage_type.num_fields

    pyarrow.register_extension_type(
        AwkwardArrowType(pyarrow.null(), None, None, None, None, None)
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

if not ak._v2._util.numpy_at_least("1.17.0"):

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


def popbuffers_finalize(out, array, validbits, awkwardarrow_type, fix_offsets=True):
    # Every buffer from Arrow must be offsets-corrected.
    if fix_offsets and (array.offset != 0 or len(array) != len(out)):
        out = out[array.offset : array.offset + len(array)]

    if isinstance(awkwardarrow_type, AwkwardArrowType):
        mask_parameters = awkwardarrow_type.mask_parameters
    else:
        mask_parameters = None

    # Everything must leave popbuffers as option-type; the mask_node will be
    # removed by the next level up in popbuffers recursion if appropriate.
    if validbits is None:
        return ak._v2.contents.UnmaskedArray(out, parameters=mask_parameters)
    else:
        return ak._v2.contents.BitMaskedArray(
            ak._v2.index.IndexU8(numpy.frombuffer(validbits, dtype=np.uint8)),
            out,
            valid_when=True,
            length=len(out),
            lsb_order=True,
            parameters=mask_parameters,
        )


def popbuffers(paarray, awkwardarrow_type, storage_type, buffers):
    # Start by removing the ExtensionArray wrapper.
    if awkwardarrow_type is not None:
        paarray = paarray.storage

    ### Beginning of the big if-elif-elif chain!

    if isinstance(storage_type, pyarrow.lib.PyExtensionType):
        raise ValueError(
            "Arrow arrays containing pickled Python objects can't be converted into Awkward Arrays"
        )

    elif isinstance(storage_type, pyarrow.lib.ExtensionType):
        # AwkwardArrowType should already be unwrapped; this must be some other ExtensionType.
        assert not isinstance(storage_type, AwkwardArrowType)
        # In that case, just ignore its logical type and use its storage type.
        return popbuffers(
            paarray, awkwardarrow_type, storage_type.storage_type, buffers
        )

    elif isinstance(storage_type, pyarrow.lib.DictionaryType):
        masked_index = popbuffers(
            paarray.indices, None, storage_type.index_type, buffers
        )
        index = masked_index.content.data

        if not isinstance(masked_index, ak._v2.contents.UnmaskedArray):
            mask = masked_index.mask_as_bool(valid_when=False)
            if mask.any():
                index = numpy.array(index, copy=True)
                index[mask] = -1

        content = handle_arrow(paarray.dictionary)

        parameters = ak._v2._util.merge_parameters(
            mask_parameters(awkwardarrow_type), node_parameters(awkwardarrow_type)
        )
        if parameters is None:
            parameters = {"__array__": "categorical"}

        return ak._v2.contents.IndexedOptionArray(
            ak._v2.index.Index(index),
            content,
            parameters=parameters,
        ).simplify_optiontype()

    elif isinstance(storage_type, pyarrow.lib.FixedSizeListType):
        assert storage_type.num_buffers == 1
        validbits = buffers.pop(0)

        a, b = to_awkwardarrow_storage_types(storage_type.value_type)
        akcontent = popbuffers(paarray.values, a, b, buffers)

        if not storage_type.value_field.nullable:
            # strip the dummy option-type node
            akcontent = remove_optiontype(akcontent)

        out = ak._v2.contents.RegularArray(
            akcontent,
            storage_type.list_size,
            parameters=node_parameters(awkwardarrow_type),
        )
        return popbuffers_finalize(out, paarray, validbits, awkwardarrow_type)

    elif isinstance(storage_type, (pyarrow.lib.LargeListType, pyarrow.lib.ListType)):
        assert storage_type.num_buffers == 2
        validbits = buffers.pop(0)
        paoffsets = buffers.pop(0)

        if isinstance(storage_type, pyarrow.lib.LargeListType):
            akoffsets = ak._v2.index.Index64(
                numpy.frombuffer(paoffsets, dtype=np.int64)
            )
        else:
            akoffsets = ak._v2.index.Index32(
                numpy.frombuffer(paoffsets, dtype=np.int32)
            )

        a, b = to_awkwardarrow_storage_types(storage_type.value_type)
        akcontent = popbuffers(paarray.values, a, b, buffers)

        if not storage_type.value_field.nullable:
            # strip the dummy option-type node
            akcontent = remove_optiontype(akcontent)

        out = ak._v2.contents.ListOffsetArray(
            akoffsets, akcontent, parameters=node_parameters(awkwardarrow_type)
        )
        return popbuffers_finalize(out, paarray, validbits, awkwardarrow_type)

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
            "Arrow arrays containing pickled Python objects can't be converted into Awkward Arrays"
        )

    elif isinstance(storage_type, pyarrow.lib.FixedSizeBinaryType):
        assert storage_type.num_buffers == 2
        validbits = buffers.pop(0)
        pacontent = buffers.pop(0)

        parameters = node_parameters(awkwardarrow_type)
        if parameters is None:
            parameters = {"__array__": "bytestring"}
        sub_parameters = {"__array__": "byte"}

        out = ak._v2.contents.RegularArray(
            ak._v2.contents.NumpyArray(
                numpy.frombuffer(pacontent, dtype=np.uint8),
                parameters=sub_parameters,
                nplike=ak.nplike.Numpy.instance(),
            ),
            storage_type.byte_width,
            parameters=parameters,
        )
        return popbuffers_finalize(out, paarray, validbits, awkwardarrow_type)

    elif storage_type in _string_like:
        assert storage_type.num_buffers == 3
        validbits = buffers.pop(0)
        paoffsets = buffers.pop(0)
        pacontent = buffers.pop(0)

        if storage_type in _string_like[::2]:
            akoffsets = ak._v2.index.Index32(
                numpy.frombuffer(paoffsets, dtype=np.int32)
            )
        else:
            akoffsets = ak._v2.index.Index64(
                numpy.frombuffer(paoffsets, dtype=np.int64)
            )

        parameters = node_parameters(awkwardarrow_type)

        if storage_type in _string_like[:2]:
            if parameters is None:
                parameters = {"__array__": "string"}
            sub_parameters = {"__array__": "char"}
        else:
            if parameters is None:
                parameters = {"__array__": "bytestring"}
            sub_parameters = {"__array__": "byte"}

        out = ak._v2.contents.ListOffsetArray(
            akoffsets,
            ak._v2.contents.NumpyArray(
                numpy.frombuffer(pacontent, dtype=np.uint8),
                parameters=sub_parameters,
                nplike=ak.nplike.Numpy.instance(),
            ),
            parameters=parameters,
        )
        return popbuffers_finalize(out, paarray, validbits, awkwardarrow_type)

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
            akcontent = popbuffers(paarray.field(field_name), a, b, buffers)
            if not field.nullable:
                # strip the dummy option-type node
                akcontent = remove_optiontype(akcontent)
            contents.append(akcontent)

        if awkwardarrow_type is not None and awkwardarrow_type.record_is_tuple:
            keys = None

        out = ak._v2.contents.RecordArray(
            contents,
            keys,
            length=len(paarray),
            parameters=node_parameters(awkwardarrow_type),
        )
        return popbuffers_finalize(
            out, paarray, validbits, awkwardarrow_type, fix_offsets=False
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
            akcontent = popbuffers(paarray.field(i), a, b, buffers)

            if not field.nullable:
                # strip the dummy option-type node
                akcontent = remove_optiontype(akcontent)
            akcontents.append(akcontent)

        out = ak._v2.contents.UnionArray(
            ak._v2.index.Index8(nptags),
            ak._v2.index.Index32(npindex),
            akcontents,
            parameters=node_parameters(awkwardarrow_type),
        )
        return popbuffers_finalize(out, paarray, None, awkwardarrow_type)

    elif storage_type == pyarrow.null():
        validbits = buffers.pop(0)
        assert storage_type.num_fields == 0

        # This is already an option-type and offsets-corrected, so no popbuffers_finalize.
        return ak._v2.contents.IndexedOptionArray(
            ak._v2.index.Index64(numpy.full(len(paarray), -1, dtype=np.int64)),
            ak._v2.contents.EmptyArray(parameters=node_parameters(awkwardarrow_type)),
            parameters=mask_parameters(awkwardarrow_type),
        )

    elif storage_type == pyarrow.bool_():
        assert storage_type.num_buffers == 2
        validbits = buffers.pop(0)
        bitdata = buffers.pop(0)

        bytedata = unpackbits(numpy.frombuffer(bitdata, dtype=np.uint8))

        out = ak._v2.contents.NumpyArray(
            bytedata.view(np.bool_),
            parameters=node_parameters(awkwardarrow_type),
            nplike=ak.nplike.Numpy.instance(),
        )
        return popbuffers_finalize(out, paarray, validbits, awkwardarrow_type)

    elif isinstance(storage_type, pyarrow.lib.DataType):
        assert storage_type.num_buffers == 2
        validbits = buffers.pop(0)
        data = buffers.pop(0)

        to64, dt = _pyarrow_to_numpy_dtype.get(str(storage_type), (False, None))
        if to64:
            data = numpy.frombuffer(data, dtype=np.int32).astype(np.int64)
        if dt is None:
            dt = storage_type.to_pandas_dtype()

        out = ak._v2.contents.NumpyArray(
            numpy.frombuffer(data, dtype=dt),
            parameters=node_parameters(awkwardarrow_type),
            nplike=ak.nplike.Numpy.instance(),
        )
        return popbuffers_finalize(out, paarray, validbits, awkwardarrow_type)

    else:
        raise TypeError("unrecognized Arrow array type: {0}".format(repr(storage_type)))


def to_awkwardarrow_type(storage_type, use_extensionarray, mask, node):
    if use_extensionarray:
        return AwkwardArrowType(
            storage_type,
            ak._v2._util.direct_Content_subclass_name(mask),
            ak._v2._util.direct_Content_subclass_name(node),
            None if mask is None else mask.parameters,
            None if node is None else node.parameters,
            node.is_tuple if isinstance(node, ak._v2.contents.RecordArray) else None,
        )
    else:
        return storage_type


def not_null(awkwardarrow_type, akarray):
    if awkwardarrow_type is None:
        # option-type but not-containing-any-nulls could accidentally be stripped
        return isinstance(akarray, ak._v2.contents.UnmaskedArray)
    else:
        # always gets option-typeness right
        return awkwardarrow_type.mask_type in (None, "IndexedArray")


def remove_optiontype(akarray):
    assert type(akarray).is_OptionType
    if isinstance(akarray, ak._v2.contents.IndexedOptionArray):
        return ak._v2.contents.IndexedArray(
            akarray.index, akarray.content, akarray.identifier, akarray.parameters
        )
    else:
        return akarray.content


def handle_arrow(obj, pass_empty_field=False):
    if isinstance(obj, pyarrow.lib.Array):
        buffers = obj.buffers()

        awkwardarrow_type, storage_type = to_awkwardarrow_storage_types(obj.type)
        out = popbuffers(obj, awkwardarrow_type, storage_type, buffers)
        assert len(buffers) == 0

        if not_null(awkwardarrow_type, out):
            return remove_optiontype(out)
        else:
            return out

    elif isinstance(obj, pyarrow.lib.ChunkedArray):
        layouts = [handle_arrow(x) for x in obj.chunks if len(x) > 0]

        if len(layouts) == 1:
            return layouts[0]
        else:
            raise NotImplementedError(
                "FIXME: need ak._v2.operations.structure.concatenate"
            )
            return ak._v2.operations.structure.concatenate(layouts, highlevel=False)

    elif isinstance(obj, pyarrow.lib.RecordBatch):
        child_array = []
        for i in range(obj.num_columns):
            layout = handle_arrow(obj.column(i))
            if obj.schema.field(i).nullable and not layout.is_OptionType:
                layout = ak._v2.contents.UnmaskedArray(layout)
            child_array.append(layout)

        if pass_empty_field and list(obj.schema.names) == [""]:
            return child_array[0]
        else:
            return ak._v2.contents.RecordArray(child_array, obj.schema.names)

    elif isinstance(obj, pyarrow.lib.Table):
        batches = obj.combine_chunks().to_batches()
        if len(batches) == 0:
            # zero-length array with the right type
            raise NotImplementedError
        elif len(batches) == 1:
            out = handle_arrow(batches[0], pass_empty_field)
        else:
            arrays = [
                handle_arrow(batch, pass_empty_field)
                for batch in batches
                if len(batch) > 0
            ]
            raise NotImplementedError(
                "FIXME: need ak._v2.operations.structure.concatenate"
            )
            out = ak._v2.operations.structure.concatenate(arrays, highlevel=False)

        if obj.schema.metadata is not None and b"ak:parameters" in obj.schema.metadata:
            optiontype, recordtype = None, None
            if out.is_OptionType:
                optiontype = out
            if out.is_RecordType:
                recordtype = out
            elif out.content.is_RecordType:
                recordtype = out.content

            parameters = json.loads(obj.schema.metadata[b"ak:parameters"])
            for x in parameters:
                (key,) = x.keys()
                (value,) = x.values()
                if key in (
                    "UnmaskedArray",
                    "BitMaskedArray",
                    "ByteMaskedArray",
                    "IndexedOptionArray",
                ):
                    optiontype._parameters = value
                elif key == "RecordArray":
                    recordtype._parameters = value

        return out

    elif (
        isinstance(obj, Iterable)
        and isinstance(obj, Sized)
        and len(obj) > 0
        and all(isinstance(x, pyarrow.lib.RecordBatch) for x in obj)
        and any(len(x) > 0 for x in obj)
    ):
        chunks = []
        for batch in obj:
            chunk = handle_arrow(batch, pass_empty_field)
            if len(chunk) > 0:
                chunks.append(chunk)
        if len(chunks) == 1:
            return chunks[0]
        else:
            raise NotImplementedError(
                "FIXME: need ak._v2.operations.structure.concatenate"
            )
            return ak._v2.operations.structure.concatenate(chunks, highlevel=False)

    elif isinstance(obj, Iterable) and len(obj) == 0:
        return ak._v2.contents.RecordArray([], [], length=0)

    else:
        raise TypeError("unrecognized Arrow type: {0}".format(type(obj)))
