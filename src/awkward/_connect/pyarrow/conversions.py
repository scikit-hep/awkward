# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import json
from collections.abc import Iterable, Sized
from functools import lru_cache

import pyarrow

import awkward as ak
from awkward._backends.numpy import NumpyBackend
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._parameters import parameters_union

from .extn_types import AwkwardArrowType, to_awkwardarrow_storage_types

np = NumpyMetadata.instance()
numpy = Numpy.instance()


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
        return pyarrow.py_buffer(numpy.packbits(validbytes, bitorder="little"))


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
                numpy.frombuffer(validbits, dtype=np.uint8)[: len(out) // 8] == 0xFF
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
                                numpy.full(
                                    -(len(out) // -8), np.uint8(0xFF), dtype=np.uint8
                                )
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
        raise ValueError(
            "Arrow arrays containing pickled Python objects can't be converted into Awkward Arrays"
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
                index = numpy.asarray(index, copy=True)
                index[mask] = -1

        content = handle_arrow(paarray.dictionary, generate_bitmasks)

        parameters = parameters_union(
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

        parameters = node_parameters(awkwardarrow_type)
        if parameters is None:
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
                backend=NumpyBackend.instance(),
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

        empty_array = ak.contents.EmptyArray(
            parameters=node_parameters(awkwardarrow_type)
        )
        if awkwardarrow_type is not None and awkwardarrow_type._is_nonnullable_nulltype:
            # Special case: pyarrow does not support a non-option null type,
            # So we short-cut the Option wrapper when _is_nonnullable_nulltype is True.
            return revertable(empty_array, empty_array)

        # This is already an option-type and offsets-corrected, so no popbuffers_finalize.
        return ak.contents.IndexedOptionArray(
            ak.index.Index64(numpy.full(len(paarray), -1, dtype=np.int64)),
            empty_array,
            parameters=mask_parameters(awkwardarrow_type),
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
            parameters=node_parameters(awkwardarrow_type),
            backend=NumpyBackend.instance(),
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
            data = numpy.astype(numpy.frombuffer(data, dtype=np.int32), dtype=np.int64)
        if dt is None:
            dt = storage_type.to_pandas_dtype()

        out = ak.contents.NumpyArray(
            numpy.frombuffer(data, dtype=dt),
            parameters=node_parameters(awkwardarrow_type),
            backend=NumpyBackend.instance(),
        )
        return popbuffers_finalize(
            out, paarray, validbits, awkwardarrow_type, generate_bitmasks
        )

    else:
        raise TypeError(f"unrecognized Arrow array type: {storage_type!r}")


def form_popbuffers(awkwardarrow_type, storage_type):
    ### Beginning of the big if-elif-elif chain!

    if isinstance(storage_type, pyarrow.lib.PyExtensionType):
        raise ValueError(
            "Arrow arrays containing pickled Python objects can't be converted into Awkward Arrays"
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
            raise TypeError(
                f"unrecognized Arrow DictionaryType index type: {index_type}"
            )

        a, b = to_awkwardarrow_storage_types(storage_type.value_type)
        content = form_popbuffers(a, b)

        parameters = parameters_union(
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
            ak.forms.EmptyForm(),
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
        raise TypeError(f"unrecognized Arrow array type: {storage_type!r}")


def to_awkwardarrow_type(
    storage_type,
    use_extensionarray,
    record_is_scalar,
    mask,
    node,
    is_nonnullable_nulltype=False,
):
    if use_extensionarray:
        return AwkwardArrowType(
            storage_type=storage_type,
            mask_type=direct_Content_subclass_name(mask),
            node_type=direct_Content_subclass_name(node),
            mask_parameters=None if mask is None else mask.parameters,
            node_parameters=None if node is None else node.parameters,
            record_is_tuple=node.is_tuple
            if isinstance(node, ak.contents.RecordArray)
            else None,
            record_is_scalar=record_is_scalar,
            is_nonnullable_nulltype=is_nonnullable_nulltype,
        )
    else:
        return storage_type


@lru_cache
def find_first_content_subclass(cls):
    for base_cls in reversed(cls.mro()):
        if base_cls is not ak.contents.Content and issubclass(
            base_cls, ak.contents.Content
        ):
            return base_cls
    raise TypeError


def direct_Content_subclass(node):
    if node is None:
        return None
    else:
        return find_first_content_subclass(type(node))


def direct_Content_subclass_name(node):
    out = direct_Content_subclass(node)
    if out is None:
        return None
    else:
        return out.__name__


def is_revertable(akarray):
    return hasattr(akarray, "__pyarrow_original")


def remove_optiontype(akarray):
    if callable(akarray.__pyarrow_original):
        return akarray.__pyarrow_original()
    else:
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
        elif any(is_revertable(arr) for arr in layouts):
            assert all(is_revertable(arr) for arr in layouts)
            # TODO: the callable argument to revertable is a premature(?) optimisation.
            #       it would be better to obviate the need to compute both revertable and non revertable branches
            #       e.g. by requesting a particular layout kind from the next `frombuffers` operation
            return revertable(
                ak.operations.concatenate(layouts, highlevel=False),
                lambda: ak.operations.concatenate(
                    [remove_optiontype(x) for x in layouts], highlevel=False
                ),
            )
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
            # create an empty array following the input schema
            return form_handle_arrow(
                obj.schema, pass_empty_field=pass_empty_field
            ).length_zero_array()
        elif len(batches) == 1:
            return handle_arrow(batches[0], generate_bitmasks, pass_empty_field)
        else:
            arrays = [
                handle_arrow(batch, generate_bitmasks, pass_empty_field)
                for batch in batches
                if len(batch) > 0
            ]
            if any(is_revertable(arr) for arr in arrays):
                assert all(is_revertable(arr) for arr in arrays)
                # TODO: the callable argument to revertable is a premature(?) optimisation.
                #       it would be better to obviate the need to compute both revertable and non revertable branches
                #       e.g. by requesting a particular layout kind from the next `frombuffers` operation
                return revertable(
                    ak.operations.concatenate(arrays, highlevel=False),
                    lambda: ak.operations.concatenate(
                        [remove_optiontype(x) for x in arrays], highlevel=False
                    ),
                )
            else:
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
        raise TypeError(f"unrecognized Arrow type: {type(obj)}")


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


def convert_to_array(layout, type=None):
    out = ak.operations.to_arrow(layout, extensionarray=False)
    if type is None:
        return out
    else:
        return pyarrow.array(out, type=type)


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
