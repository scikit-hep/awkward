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

### FIXME!!!
# else:
#     if distutils.version.LooseVersion(
#         pyarrow.__version__
#     ) < distutils.version.LooseVersion("6.0.0"):
#         pyarrow = None
#         error_message = "pyarrow 6.0.0 or later required for {0}"


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
        pass

    class AwkwardArrowType(pyarrow.ExtensionType):
        def __init__(
            self,
            storage_type,
            mask_type,
            node_type,
            mask_parameters,
            node_parameters,
        ):
            self._mask_type = mask_type
            self._node_type = node_type
            self._mask_parameters = mask_parameters
            self._node_parameters = node_parameters
            super(AwkwardArrowType, self).__init__(storage_type, "awkward")

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

        def __arrow_ext_class__(self):
            return AwkwardArrowArray

        def __arrow_ext_serialize__(self):
            return json.dumps(
                {
                    "mask_type": self._mask_type,
                    "node_type": self._node_type,
                    "mask_parameters": self._mask_parameters,
                    "node_parameters": self._node_parameters,
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
            )

    pyarrow.register_extension_type(
        AwkwardArrowType(pyarrow.null(), None, None, None, None)
    )


if distutils.version.LooseVersion(numpy.__version__) < distutils.version.LooseVersion(
    "1.17.0"
):

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

    def unpackbits(bitarray, length, lsb_order=True):
        ready_to_bitswap = numpy.unpackbits(bitarray)
        if lsb_order:
            return ready_to_bitswap.reshape(-1, 8)[:, ::-1].reshape(-1)[:length]
        else:
            return ready_to_bitswap[:length]


else:

    def packbits(bytearray, lsb_order=True):
        return numpy.packbits(bytearray, bitorder=("little" if lsb_order else "big"))

    def unpackbits(bitarray, length, lsb_order=True):
        return numpy.unpackbits(
            bitarray, count=length, bitorder=("little" if lsb_order else "big")
        )


def to_validbits(bytemask, valid_when):
    if bytemask is None:
        return None
    else:
        out = packbits(bytemask.view(np.bool_))
        if not valid_when:
            numpy.bitwise_not(out, out=out)
        return out


def and_validbits(validbits1, validbits2):
    if validbits1 is None:
        return validbits2
    elif validbits2 is None:
        return validbits1
    else:
        return validbits1 & validbits2


pyarrow_to_numpy_dtype = {
    "date32": (True, np.dtype("M8[D]")),
    "date64": (False, np.dtype("M8[ms]")),
    "date32[day]": (True, np.dtype("M8[D]")),
    "date64[ms]": (False, np.dtype("M8[ms]")),
    "time32[s]": (True, np.dtype("M8[s]")),
    "time32[ms]": (True, np.dtype("M8[ms]")),
    "time64[us]": (False, np.dtype("M8[us]")),
    "time64[ns]": (False, np.dtype("M8[ns]")),
    "timestamp[s]": (False, np.dtype("M8[s]")),
    "timestamp[ms]": (False, np.dtype("M8[ms]")),
    "timestamp[us]": (False, np.dtype("M8[us]")),
    "timestamp[ns]": (False, np.dtype("M8[ns]")),
    "duration[s]": (False, np.dtype("m8[s]")),
    "duration[ms]": (False, np.dtype("m8[ms]")),
    "duration[us]": (False, np.dtype("m8[us]")),
    "duration[ns]": (False, np.dtype("m8[ns]")),
}


def popbuffers(array, extension_type, storage_type, buffers):
    if isinstance(storage_type, pyarrow.lib.ExtensionType):
        return popbuffers(array, storage_type, storage_type.storage_type, buffers)

    elif isinstance(storage_type, pyarrow.lib.PyExtensionType):
        raise NotImplementedError

    elif isinstance(storage_type, pyarrow.lib.DictionaryType):
        raise NotImplementedError

    elif isinstance(storage_type, pyarrow.lib.FixedSizeListType):
        raise NotImplementedError

    elif isinstance(storage_type, pyarrow.lib.LargeListType):
        raise NotImplementedError

    elif isinstance(storage_type, pyarrow.lib.ListType):
        raise NotImplementedError

    elif isinstance(storage_type, pyarrow.lib.MapType):
        raise NotImplementedError

    elif isinstance(storage_type, pyarrow.lib.FixedSizeBinaryType):
        raise NotImplementedError

    elif storage_type == pyarrow.large_string():
        raise NotImplementedError

    elif storage_type == pyarrow.string():
        raise NotImplementedError

    elif storage_type == pyarrow.large_binary():
        raise NotImplementedError

    elif storage_type == pyarrow.binary():
        raise NotImplementedError

    elif isinstance(storage_type, pyarrow.lib.StructType):
        raise NotImplementedError

    elif isinstance(storage_type, pyarrow.lib.DenseUnionType):
        raise NotImplementedError

    elif isinstance(storage_type, pyarrow.lib.SparseUnionType):
        raise NotImplementedError

    elif isinstance(storage_type, pyarrow.lib.TimestampType):
        raise NotImplementedError

    elif isinstance(storage_type, pyarrow.lib.Time64Type):
        raise NotImplementedError

    elif isinstance(storage_type, pyarrow.lib.Time32Type):
        raise NotImplementedError

    elif isinstance(storage_type, pyarrow.lib.DurationType):
        raise NotImplementedError

    elif isinstance(storage_type, pyarrow.lib.Decimal256Type):
        raise NotImplementedError

    elif isinstance(storage_type, pyarrow.lib.Decimal128Type):
        raise NotImplementedError

    elif storage_type == pyarrow.bool_():
        assert storage_type.num_buffers == 2
        mask = buffers.pop(0)
        data = buffers.pop(0)

        bytearray = unpackbits(numpy.frombuffer(data, dtype=np.uint8), len(array))

        parameters = None
        if isinstance(extension_type, AwkwardArrowType):
            parameters = extension_type.node_parameters

        out = ak._v2.contents.NumpyArray(
            bytearray.view(np.bool_), parameters=parameters
        )
        # no return yet!

    elif (
        isinstance(storage_type, pyarrow.lib.DataType) and storage_type.num_buffers == 1
    ):
        # this is DataType(null)
        mask = buffers.pop(0)
        assert storage_type.num_fields == 0

        node_parameters, mask_parameters = None, None
        if isinstance(extension_type, AwkwardArrowType):
            node_parameters = extension_type.node_parameters
            mask_parameters = extension_type.mask_parameters

        out = ak._v2.contents.IndexedOptionArray(
            ak._v2.index.Index64(numpy.full(len(array), -1, dtype=np.int64)),
            ak._v2.contents.EmptyArray(parameters=node_parameters),
            parameters=mask_parameters,
        )

        if array.offset == 0 and len(array) == len(out):
            return out
        else:
            return out[array.offset : array.offset + len(array)]
        # RETURNED an option-type array with offsets corrected

    elif isinstance(storage_type, pyarrow.lib.DataType):
        assert storage_type.num_buffers == 2
        mask = buffers.pop(0)
        data = buffers.pop(0)

        to64, dt = pyarrow_to_numpy_dtype.get(str(storage_type), (False, None))
        if to64:
            data = numpy.frombuffer(data, dtype=np.int32).astype(np.int64)
        if dt is None:
            dt = storage_type.to_pandas_dtype()

        parameters = None
        if isinstance(extension_type, AwkwardArrowType):
            parameters = extension_type.node_parameters

        out = ak._v2.contents.NumpyArray(
            numpy.frombuffer(data, dtype=dt), parameters=parameters
        )
        # no return yet!

    else:
        raise TypeError("unrecognized Arrow array type: {0}".format(repr(storage_type)))

    parameters = None
    if isinstance(extension_type, AwkwardArrowType):
        parameters = extension_type.mask_parameters

    # All 'no return yet' cases need to become option-type (even if the UnmaskedArray
    # is just going to get stripped off in the recursive step that calls this one).
    if mask is None:
        out = ak._v2.contents.UnmaskedArray(out, parameters=parameters)
    else:
        mask = ak._v2.index.IndexU8(numpy.frombuffer(mask, dtype=np.uint8))
        out = ak._v2.contents.BitMaskedArray(
            mask,
            out,
            valid_when=True,
            length=len(out),
            lsb_order=True,
            parameters=parameters,
        )

    # All 'no return yet' cases need to be corrected for pyarrow's 'offset'.
    if array.offset == 0 and len(array) == len(out):
        return out
    else:
        return out[array.offset : array.offset + len(array)]


def extension_storage(tpe):
    if isinstance(tpe, pyarrow.lib.ExtensionType):
        return tpe, tpe.storage_type
    else:
        return None, tpe


def not_null(extension_type, array):
    if isinstance(extension_type, AwkwardArrowType):
        return extension_type.mask_type is None
    else:
        return isinstance(array, ak._v2.contents.UnmaskedArray)


def handle_arrow(obj, pass_empty_field=True):
    if isinstance(obj, pyarrow.lib.Array):
        buffers = obj.buffers()
        extension_type, storage_type = extension_storage(obj.type)

        out = popbuffers(obj, extension_type, storage_type, buffers)
        assert len(buffers) == 0

        if not_null(extension_type, out):
            assert type(out).is_OptionType
            return out.content
        else:
            return out

    elif isinstance(obj, pyarrow.lib.ChunkedArray):
        layouts = [handle_arrow(x, pass_empty_field) for x in obj.chunks if len(x) > 0]

        if len(layouts) == 1:
            return layouts[0]
        else:
            return ak._v2.operations.structure.concatenate(layouts, highlevel=False)

    elif isinstance(obj, pyarrow.lib.RecordBatch):
        child_array = []
        for i in range(obj.num_columns):
            layout = handle_arrow(obj.column(i), pass_empty_field)
            if obj.schema.field(i).nullable and not isinstance(
                layout, ak._v2._util.optiontypes
            ):
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
            return handle_arrow(batches[0], pass_empty_field)
        else:
            arrays = [
                handle_arrow(batch, pass_empty_field)
                for batch in batches
                if len(batch) > 0
            ]
            return ak._v2.operations.structure.concatenate(arrays, highlevel=False)

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
            return ak._v2.operations.structure.concatenate(chunks, highlevel=False)

    else:
        raise TypeError("unrecognized Arrow type: {0}".format(type(obj)))
