import awkward1
import awkward1.highlevel
import numpy as np
import sys
import codecs
import math

import collections
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


def from_iter(iterable,
              highlevel=True,
              behavior=None,
              allowrecord=True,
              initial=1024,
              resize=1.5):
    """
    Args:
        iterable (Python iterable): Data to convert into an Awkward Array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (bool): Custom #ak.behavior for the output array, if
            high-level.
        allowrecord (bool): If True, the outermost element may be a record
            (returning #ak.Record or #ak.layout.Record type, depending on
            `highlevel`); if False, the outermost element must be an array.
        initial (int): Initial size (in bytes) of buffers used by
            # ak.layout.ArrayBuilder (see #ak.layout.ArrayBuilderOptions).
        resize (float): Resize multiplier for buffers used by
            # ak.layout.ArrayBuilder (see #ak.layout.ArrayBuilderOptions);
            should be strictly greater than 1.

    Converts Python data into an Awkward Array.

    Internally, this function uses #ak.layout.ArrayBuilder (see the high-level
    # ak.ArrayBuilder documentation for a more complete description), so it
    has the same flexibility and the same constraints. Any heterogeneous
    and deeply nested Python data can be converted, but the output will never
    have regular-typed array lengths.

    The following Python types are supported.

       * bool, including `np.bool_`: converted into #ak.layout.NumpyArray.
       * int, including `np.integer`: converted into #ak.layout.NumpyArray.
       * float, including `np.floating`: converted into #ak.layout.NumpyArray.
       * bytes: converted into #ak.layout.ListOffsetArray with parameter
         `"__array__"` equal to `"bytestring"` (unencoded bytes).
       * str: converted into #ak.layout.ListOffsetArray with parameter
         `"__array__"` equal to `"string"` (UTF-8 encoded string).
       * tuple: converted into #ak.layout.RecordArray without field names
         (i.e. homogeneously typed, uniform sized tuples).
       * dict: converted into #ak.layout.RecordArray with field names
         (i.e. homogeneously typed records with the same sets of fields).
       * iterable, including np.ndarray: converted into
         # ak.layout.ListOffsetArray.

    See also #ak.to_list.
    """
    if isinstance(iterable, dict):
        if allowrecord:
            return from_iter([iterable],
                             highlevel=highlevel,
                             behavior=behavior,
                             initial=initial,
                             resize=resize)[0]
        else:
            raise ValueError("cannot produce an array from a dict")
    out = awkward1.layout.ArrayBuilder(initial=initial, resize=resize)
    for x in iterable:
        out.fromiter(x)
    layout = out.snapshot()
    if highlevel:
        return awkward1._util.wrap(layout, behavior)
    else:
        return layout


def to_layout(array,
              allowrecord=True,
              allowother=False,
              numpytype=(np.number, )):
    """
    Args:
        array: Data to convert into an #ak.layout.Content and maybe
            # ak.layout.Record and other types.
        allowrecord (bool): If True, allow #ak.layout.Record as an output;
            otherwise, if the output would be a scalar record, raise an error.
        allowother (bool): If True, allow non-Awkward outputs; otherwise,
            if the output would be another type, raise an error.
        numpytype (tuple of np types): Allowed np types in
            # ak.layout.NumpyArray outputs.

    Converts `array` (many types supported, including all Awkward Arrays and
    Records) into a #ak.layout.Content and maybe #ak.layout.Record and other
    types.

    This function is usually used to sanitize inputs for other functions; it
    would rarely be used in a data analysis.
    """
    import awkward1.highlevel

    if isinstance(array, awkward1.highlevel.Array):
        return array.layout

    elif allowrecord and isinstance(array, awkward1.highlevel.Record):
        return array.layout

    elif isinstance(array, awkward1.highlevel.ArrayBuilder):
        return array.snapshot().layout

    elif isinstance(array, awkward1.layout.ArrayBuilder):
        return array.snapshot()

    elif isinstance(array, awkward1.layout.Content):
        return array

    elif allowrecord and isinstance(array, awkward1.layout.Record):
        return array

    elif isinstance(array, np.ma.MaskedArray):
        mask = np.ma.getmask(array)
        data = np.ma.getdata(array)
        if mask is False:
            out = awkward1.layout.UnmaskedArray(
                awkward1.layout.NumpyArray(data.reshape(-1)))
        else:
            out = awkward1.layout.ByteMaskedArray(
                awkward1.layout.Index8(mask.reshape(-1)),
                awkward1.layout.NumpyArray(data.reshape(-1)))
        for size in array.shape[:0:-1]:
            out = awkward1.layout.RegularArray(out, size)
        return out

    elif isinstance(array, np.ndarray):
        if not issubclass(array.dtype.type, numpytype):
            raise ValueError("np {0} not allowed".format(repr(array.dtype)))
        out = awkward1.layout.NumpyArray(array.reshape(-1))
        for size in array.shape[:0:-1]:
            out = awkward1.layout.RegularArray(out, size)
        return out

    elif (isinstance(array, (str, bytes))
          or (awkward1._util.py27 and isinstance(array, unicode))):
        return from_iter([array], highlevel=False)

    elif isinstance(array, Iterable):
        return from_iter(array, highlevel=False)

    elif not allowother:
        raise TypeError(
            "{0} cannot be converted into an Awkward Array".format(array))

    else:
        return array


def to_arrow(layout):
    import pyarrow
    import math

    def recurse(layout, mask=None):

        if isinstance(layout, awkward1.layout.NumpyArray):
            numpy_arr = np.asarray(layout)
            if (numpy_arr.ndim == 1):
                if mask is not None:
                    return pyarrow.Array.from_buffers(pyarrow.from_numpy_dtype(numpy_arr.dtype), len(numpy_arr), [pyarrow.py_buffer(mask), pyarrow.py_buffer(numpy_arr)])
                else:
                    return pyarrow.Array.from_buffers(pyarrow.from_numpy_dtype(numpy_arr.dtype), len(numpy_arr), [None, pyarrow.py_buffer(numpy_arr)])
            else:
                return pyarrow.Tensor.from_numpy(numpy_arr)

        elif isinstance(layout, awkward1.layout.EmptyArray):
            return pyarrow.Array.from_buffers(pyarrow.float64(), 0, [None, None])

        elif isinstance(layout, (awkward1.layout.IndexU8,
                                 awkward1.layout.IndexU32,
                                 awkward1.layout.Index8,
                                 awkward1.layout.Index32,
                                 awkward1.layout.Index64)):

            numpy_arr = np.asarray(layout)
            if mask is not None:
                return pyarrow.Array.from_buffers(pyarrow.from_numpy_dtype(numpy_arr.dtype), len(numpy_arr), [pyarrow.py_buffer(mask), pyarrow.py_buffer(numpy_arr)])
            else:
                return pyarrow.Array.from_buffers(pyarrow.from_numpy_dtype(numpy_arr.dtype), len(numpy_arr), [None, pyarrow.py_buffer(numpy_arr)])

        elif isinstance(layout, awkward1.layout.ListOffsetArray32):

            offsets = np.asarray(layout.offsets, dtype=np.int32)

            if (awkward1.operations.describe.parameters(layout).get("__array__") == "bytestring"):
                if mask is None:
                    arrow_arr = pyarrow.BinaryArray.from_buffers(
                        len(offsets) - 1, pyarrow.py_buffer(offsets), pyarrow.py_buffer(layout.content))
                else:
                    arrow_arr = pyarrow.BinaryArray.from_buffers(
                        len(offsets) - 1, pyarrow.py_buffer(offsets), pyarrow.py_buffer(layout.content), pyarrow.py_buffer(mask))

                return arrow_arr

            if (awkward1.operations.describe.parameters(layout).get("__array__") == "string"):
                if mask is None:
                    arrow_arr = pyarrow.StringArray.from_buffers(
                        len(offsets) - 1, pyarrow.py_buffer(offsets), pyarrow.py_buffer(layout.content))
                else:
                    arrow_arr = pyarrow.StringArray.from_buffers(
                        len(offsets) - 1, pyarrow.py_buffer(offsets), pyarrow.py_buffer(layout.content), pyarrow.py_buffer(mask))

                return arrow_arr

            content_buffer = recurse(layout.content)
            if mask is None:
                arrow_arr = pyarrow.Array.from_buffers(pyarrow.list_(content_buffer.type), len(
                    offsets) - 1, [None, pyarrow.py_buffer(offsets)], children=[content_buffer])
            else:
                arrow_arr = pyarrow.Array.from_buffers(pyarrow.list_(content_buffer.type), len(
                    offsets) - 1, [pyarrow.py_buffer(mask), pyarrow.py_buffer(offsets)], children=[content_buffer])

            return arrow_arr

        elif isinstance(layout, (awkward1.layout.ListOffsetArray64,
                                 awkward1.layout.ListOffsetArrayU32)):

            offsets = np.asarray(layout.offsets, dtype=np.int64)

            if (awkward1.operations.describe.parameters(layout).get("__array__") == "bytestring"):
                if mask is None:
                    arrow_arr = pyarrow.LargeBinaryArray.from_buffers(
                        len(offsets) - 1, pyarrow.py_buffer(offsets), pyarrow.py_buffer(layout.content))
                else:
                    arrow_arr = pyarrow.LargeBinaryArray.from_buffers(
                        len(offsets) - 1, pyarrow.py_buffer(offsets), pyarrow.py_buffer(layout.content), pyarrow.py_buffer(mask))

                return arrow_arr

            if (awkward1.operations.describe.parameters(layout).get("__array__") == "string"):
                if mask is None:
                    arrow_arr = pyarrow.LargeStringArray.from_buffers(
                        len(offsets) - 1, pyarrow.py_buffer(offsets), pyarrow.py_buffer(layout.content))
                else:
                    arrow_arr = pyarrow.LargeStringArray.from_buffers(
                        len(offsets) - 1, pyarrow.py_buffer(offsets), pyarrow.py_buffer(layout.content), pyarrow.py_buffer(mask))

                return arrow_arr

            content_buffer = recurse(layout.content)
            if mask is None:
                arrow_arr = pyarrow.Array.from_buffers(pyarrow.large_list(content_buffer.type), len(
                    offsets) - 1, [None, pyarrow.py_buffer(offsets)], children=[content_buffer])
            else:
                arrow_arr = pyarrow.Array.from_buffers(pyarrow.large_list(content_buffer.type), len(
                    offsets) - 1, [pyarrow.py_buffer(mask), pyarrow.py_buffer(offsets)], children=[content_buffer])

            return arrow_arr

        elif isinstance(layout, (awkward1.layout.RegularArray)):
            if mask is not None:
                return recurse(layout.broadcast_tooffsets64(layout.compact_offsets64()), mask)
            else:
                return recurse(layout.broadcast_tooffsets64(layout.compact_offsets64()))

        elif isinstance(layout, (awkward1.layout.ListArray32,
                                 awkward1.layout.ListArrayU32,
                                 awkward1.layout.ListArray64)):
            if mask is not None:
                return recurse(layout.broadcast_tooffsets64(layout.compact_offsets64()), mask)
            else:
                return recurse(layout.broadcast_tooffsets64(layout.compact_offsets64()))

        elif isinstance(layout, (awkward1.layout.RecordArray)):
            values = [recurse(x) for x in layout.contents]

            min_list_len = min(map(len, values))

            types = pyarrow.struct(
                [pyarrow.field(layout.keys()[i], values[i].type) for i in range(len(values))])

            if mask is not None:
                return pyarrow.Array.from_buffers(types, min_list_len, [pyarrow.py_buffer(mask)], children=values)
            else:
                return pyarrow.Array.from_buffers(types, min_list_len, [None], children=values)

        elif isinstance(layout, (awkward1.layout.UnionArray8_32,
                                 awkward1.layout.UnionArray8_64,
                                 awkward1.layout.UnionArray8_U32)):

            values = [to_arrow(x) for x in layout.contents]
            types = pyarrow.union(pyarrow.struct(
                [pyarrow.field(str(i), values[i].type) for i in range(len(values))]), "dense", [x for x in range(len(values))])

            if mask is not None:
                return pyarrow.Array.from_buffers(types, len(layout.tags), [pyarrow.py_buffer(mask), pyarrow.py_buffer(np.asarray(layout.tags)), pyarrow.py_buffer(np.asarray(layout.index))], children=values)
            else:
                return pyarrow.Array.from_buffers(types, len(layout.tags), [None, pyarrow.py_buffer(np.asarray(layout.tags)), pyarrow.py_buffer(np.asarray(layout.index))], children=values)

        elif isinstance(layout, (awkward1.layout.IndexedArray32,
                                 awkward1.layout.IndexedArrayU32,
                                 awkward1.layout.IndexedArray64)):

            if mask is not None:
                index = np.asarray(layout.index)
                content_buffer = recurse(layout.content[layout.index])
                print(content_buffer.buffers())
                dic_type = pyarrow.dictionary(pyarrow.from_numpy_dtype(index.dtype), content_buffer.type)
                arrow_arr = pyarrow.DictionaryArray.from_buffers(dic_type, len(index), [None, content_buffer.buffers()])
                
                ValueError("Mask in pyarrow not implemented yet")
            else:
                arrow_arr = pyarrow.DictionaryArray.from_arrays(
                    recurse(layout.index), recurse(layout.content))

            return arrow_arr

        elif isinstance(layout, (awkward1.layout.IndexedOptionArray32)):
            # Convert this to a Indexed Array
            index = np.asarray(layout.index)
            for i in range(len(index)):
                if index[i] < 0:
                    index[i] = 0
            index = awkward1.layout.Index32(index)
            array = awkward1.layout.IndexedArray32(index, layout.content)
            return recurse(array, mask)

        elif isinstance(layout, (awkward1.layout.IndexedOptionArray64)):
            # Convert this to a Indexed Array
            index = np.asarray(layout.index)
            for i in range(len(index)):
                if index[i] < 0:
                    index[i] = 0
            index = awkward1.layout.Index64(index)
            array = awkward1.layout.IndexedArray64(index, layout.content)
            return recurse(array, mask)

        elif isinstance(layout, (awkward1.layout.BitMaskedArray)):
            bitmask = np.asarray(layout.mask, dtype=np.uint8)

            if layout.lsb_order is False:
                bitmask = np.packbits(np.unpackbits(
                    bitmask).reshape(-1, 8)[:, ::-1].reshape(-1))

            if layout.valid_when is False:
                bitmask = ~bitmask

            return recurse(layout.content, bitmask)

        elif isinstance(layout, (awkward1.layout.ByteMaskedArray)):
            mask = np.asarray(
                layout.mask, dtype=np.bool) == layout.valid_when

            bytemask = np.zeros(
                8 * math.ceil(len(layout.content) / 8), dtype=np.bool)
            bytemask[:len(mask)] = mask
            bitmask = np.packbits(
                bytemask.reshape(-1, 8)[:, ::-1].reshape(-1))

            return recurse(layout.content, bitmask).slice(length=len(mask))

        elif isinstance(layout, (awkward1.layout.UnmaskedArray)):
            return recurse(layout.content)

    return recurse(layout)


def from_arrow(array):

    def recurse(array, mask=None):

        def popbuffers(array, tpe, buffers, length):
            if isinstance(tpe, pa.lib.DictionaryType):
                index = popbuffers(None if array is None else array.indices, tpe.index_type, buffers, length)
                if hasattr(tpe, "dictionary"):
                    content = fromarrow(tpe.dictionary)
                elif array is not None:
                    content = fromarrow(array.dictionary)
                else:
                    raise NotImplementedError("no way to access Arrow dictionary inside of UnionArray")
                if isinstance(index, awkwardlib.BitMaskedArray):
                    return awkwardlib.BitMaskedArray(index.mask, awkwardlib.IndexedArray(index.content, content), maskedwhen=index.maskedwhen, lsborder=index.lsborder)
                else:
                    return awkwardlib.IndexedArray(index, content)
            
            elif isinstance(tpe, pa.lib.DataType):
                assert getattr(tpe, "num_buffers", 2) == 2
                out = awkward1.layout.NumpyArray(array.to_numpy()) 
                return out
        if isinstance(array, pa.Tensor):
            return(awkward1.layout.NumpyArray(array.to_numpy()))
        
        elif isinstance(array, pa.LargeListArray):
            offsets = awkward1.layout.Index64(array.offsets.to_numpy())
            contents = recurse(array.values)
            return awkward1.layout.ListOffsetArray64(offsets, contents)
       
        elif isinstance(array, pa.StructArray):
            child_array = [recurse(array.field(x)) for x in range(array.type.num_children)]
            keys = [array.type[x].name for x in range(array.type.num_children)]
            return awkward1.layout.RecordArray(child_array, keys)
            

        elif isinstance(array, pa.lib.Array):
            buffers = array.buffers()
            out = popbuffers(array, array.type, buffers, len(array))
            return out

        elif isinstance(obj, pyarrow.lib.ChunkedArray):
            chunks = [x for x in obj.chunks if len(x) > 0]
            if len(chunks) == 1:
                return fromarrow(chunks[0])
            else:
                return awkwardlib.ChunkedArray([fromarrow(x) for x in chunks], chunksizes=[len(x) for x in chunks])

        elif isinstance(obj, pyarrow.lib.RecordBatch):
            out = awkwardlib.Table()
            for n, x in zip(obj.schema.names, obj.columns):
                out[n] = fromarrow(x)
            return out

        elif isinstance(obj, pyarrow.lib.Table):
            chunks = []
            chunksizes = []
            for batch in obj.to_batches():
                chunk = fromarrow(batch)
                if len(chunk) > 0:
                    chunks.append(chunk)
                    chunksizes.append(len(chunk))
            if len(chunks) == 1:
                return chunks[0]
            else:
                return awkwardlib.ChunkedArray(chunks, chunksizes=chunksizes)

        else:
            raise NotImplementedError(type(obj))
    
    return recurse(array)

def from_arrow_1(array):
    import pyarrow
    import numpy
    import awkward

    # print(array.type.value_type)
    if isinstance(array, pyarrow.LargeStringArray):
        print("Reached")
        mask = array.buffers()[0]
        length = len(array)
        print(array.buffers())
        offsets = numpy.frombuffer(array.buffers()[1], dtype=numpy.int64)
        contents = numpy.frombuffer(array.buffers()[2], dtype=numpy.uint8)
        # offsets = numpy.frombuffer(array.value_offsets)
        # content = numpy.frombuffer(array.buffers()[4], dtype=numpy.uint8)

        offsets = awkward1.layout.Index64(offsets)
        contents =  awkward1.layout.NumpyArray(contents)
        contents.setparameter("__array__", "char")
        awk_arr = awkward1.layout.ListOffsetArray64(offsets, contents)
        awk_arr.setparameter("__array__", "string") 
        print(awkward1.to_list(awk_arr))
        # print("Hello")
        # if mask is None:
        #     return awk_arr
        # else:
        #     awk_mask = awkward1.layout.IndexU8(numpy.frombuffer(mask.to_pybytes(), dtype=numpy.uint8))
        #     return awkward1.layout.BitMaskedArray(awk_mask, awk_arr, True, len(offsets) - 1, True)


def convert():
    # content = awkward1.layout.NumpyArray(
    #     np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10]))
    # # array = awkward1.layout.NumpyArray(
    # #     np.array([]))
    
    # offsets = awkward1.layout.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    # listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
    # print(to_arrow(listoffsetarray).flatten())
    # print(to_arrow(listoffsetarray).values)
    # print(listoffsetarray)
    # print(to_arrow(listoffsetarray).type)
    # print(from_arrow(to_arrow(listoffsetarray)))
    # regulararray = awkward1.layout.RegularArray(listoffsetarray, 2)
    # # # starts = awkward1.layout.Index64(np.array([0, 1]))
    # # # stops = awkward1.layout.Index64(np.array([2, 3]))
    # # # listarray = awkward1.layout.ListArray64(starts, stops, regulararray)
    # # # # emptyarray = awkward1.layout.EmptyArray()
    # content1 = awkward1.layout.NumpyArray(np.array([1, 2, 3, 4, 5]))
    # content2 = awkward1.layout.NumpyArray(
    #     np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    # offsets = awkward1.layout.Index32(np.array([0, 3, 3, 5, 6, 9]))

    # recordarray = awkward1.layout.RecordArray(
    #     [content1, listoffsetarray, content2, content1], keys=["one", "chonks", "2", "wonky"])
    # # print(to_arrow(recordarray).dictionary_encode())
    # print(from_arrow(to_arrow(recordarray)))








    # content0 = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
    # content = awkward1.Array(
    #     ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]).layout
    # # listoffsetarray = awkward1.layout.ListOffsetArray32(offsets, content)
    # tags = awkward1.layout.Index8(
    #     np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))
    # index = awkward1.layout.Index32(
    #     np.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=np.int32))
    # array = awkward1.layout.UnionArray8_32(tags, index, [content0, content])
    # print(to_arrow(array).buffers())


















    # # print(to_arrow(array))
    # # regulararray = awkward1.layout.RegularArray(array, 2)
    # #
    # content = awkward1.layout.NumpyArray(
    #     np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    # index = awkward1.layout.Index64(
    #     np.array([0, 2, 4, 6, 8, 9, 7, 5], dtype=np.int64))
    # indexedarray = awkward1.layout.IndexedArray64(index, content)
    # # BitMaskedArray = awkward1.layout.BitMaskedArray(
    # #     awkward1.layout.IndexU8(
    # #         np.array([40, 173, 59, 104, 182, 116], dtype=np.uint8)),
    # #     awkward1.layout.NumpyArray(
    # #         np.array([
    # #             5.5, 6.6, 1.5, 3.2, 9.8, 0.4, 5.7, 1.5, 0.2, 6.1, 5.4, 4.3,
    # #             5.9, 10.1, -2.3, 5.8, 3.4, 5.6, 6.2, 8.8, 3.1, 7.0, 1.2, 7.3,
    # #             5.8, 8.3, 9.7, 5.2, 3.4, 5.8, 1.7, 4.3, 5.8, 1.2, 1.7, 3.6,
    # #             4.4, 9.7, 5.0, 4.3, 7.8, 6.1, 3.3, 7.9, 7.1, 6.5, -0.6, 8.2,
    # #             3.7, 4.6, 3.9, 7.5
    # #         ])), False, 46, False)

    # # # bytemaskedarray = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(np.array([True, True, False, False, False], dtype=np.int8)), content, True)
    # # # print(listoffsetarray)
    # # # print(awkward1.operations.describe.parameters(listoffsetarray.content))
    # # # print(awkwardtolist(listoffsetarray))
    # # # print(listoffsetarray.__getattribute__.__array__)
    # # # print(len(BitMaskedArray.parameter.))
    # # # layout = to_layout(listoffsetarray,
    # # #                    allowrecord=True,
    # # #                    allowother=False,
    # # #                    numpytype=(np.generic,))

    # # # print(indexedarray.content[indexedarray.index])
    # # # mask = np.asarray(BitMaskedArray.mask)
    # # # np.unpackbits(mask)

    # # # if BitMaskedArray.lsb_order is False:

    # # # print()
    # # # content = awkward1.Array([
    # # #     "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
    # # # ]).layout
    # # # bytemask = awkward1.layout.Index8(np.array([False, True, False], dtype=np.bool))

    # # # array = awkward1.layout.ByteMaskedArray(bytemask, content, True)
    # # # # assert awkward1.to_arrow(array).to_pylist() == awkward1.to_list(array)
    # # # print(awkward1.tolist(array))
    # # # print(array)
    # # # print(to_arrow(array))

    # # content = awkward1.layout.NumpyArray(
    # #     np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10]))
    # # offsets = awkward1.layout.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    # # listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
    # # array = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(
    # #     np.array([True, True, False, False, False], dtype=np.int8)), content, True)
    # # offsets1 = awkward1.layout.Index64(np.array([0, 1, 3, 5]))
    # # listoffsetarray1 = awkward1.layout.ListOffsetArray64(
    # #     offsets1, listoffsetarray)
    # # # print(awkward1.tolist(array))
    # # # print(array)
    # # ["1", "2"]
    # # dic = {"": ""}
    # # # print(listoffsetarray.type(dic))
    # # # print(dic)
    # print(isinstance(to_arrow(indexedarray).type, pa.lib.DictionaryType))
    # print(to_arrow(indexedarray).indices)
    # print(to_arrow(indexedarray).buffers())import awkward1
    import pyarrow
    import numpy
    content = awkward1.Array(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]).layout
    # offsets = awkward1.layout.Index32(numpy.array([0, 3, 3, 5, 6, 9]))
    # array = awkward1.layout.ListOffsetArray32(offsets, content)
    print(content)
    print(from_arrow_1(to_arrow(content)))


if __name__ == "__main__":
    convert()
    pass

# ty = pa.struct([pa.field('a', pa.int16()), pa.field('b', pa.utf8())])
# array = pa.array([{'bee': 3, 'b': 'foo'}, None, None,
#                   {'bee': 16, 'b': 'bamboo'}], type=ty)
# buffers = array.buffers()
# buffers[:1]
# buffers
# pa.BufferReader(buffers[0]).read()
# pa.BufferReader(buffers[1]).read()
# pa.BufferReader(buffers[2]).read()
# pa.BufferReader(buffers[3]).read()
# pa.BufferReader(buffers[4]).read()
# pa.BufferReader(buffers[5]).read()


# import pyarrow as pa
# arr1 = pa.array([1, 2, 3, 4, 5])

# arr2 = pa.array([1.1, 2.2, 3.3, 4.4, 5.5])
# arr2.buffers()
# types = pa.array([0, 1, 0, 0, 1, 1, 0], type='int8')
# types.buffers()
# value_offsets = pa.array([1, 0, 0, 2, 1, 2, 3], type='int32')
# value_offsets.buffers()

# arr = pa.UnionArray.from_dense(types, value_offsets,
#                                [arr1, arr2])

# type_arr = pa.union(pa.struct(
#     [pa.field("0", arr1.type), pa.field("1", arr2.type)]), "dense", [0, 1])

# arr.buffers()

# arr4 = pa.Array.from_buffers(type_arr, 5, arr.buffers()[
#                              0:3], children=[arr1, arr2])
