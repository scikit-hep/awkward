# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import numba
import numba.core.typing.npydecl
import numpy as np

import awkward as ak
from awkward._connect.numba.growablebuffer import GrowableBuffer, GrowableBufferType
from awkward._nplikes.numpylike import ArrayLike
from awkward._typing import final


class LayoutBuilder:
    def __init__(self, content, *, parameters=None):
        self._content = content
        self._parameters = parameters

    # def get_item(self) -> builder
    #     ...

    @property
    def type(self):
        return self._type({})

    @property
    def content(self):
        return self._content

    def clear(self):
        return self._content.clear()

    def is_valid(self, error: str):
        return self._content.is_valid(error)

    def _type(self, typestrs):
        raise NotImplementedError


def tonumbatype(content):
    if isinstance(content, Numpy):
        return Numpy.numbatype(content)
    if isinstance(content, Empty):
        return Empty.numbatype(content)
    if isinstance(content, ListOffset):
        return ListOffset.numbatype(content)


class LayoutBuilderType(numba.types.Type):
    def __init__(self):
        super().__init__(name="ak.numba.lb.LayoutBuilder()")

    @property
    def parameters(self):
        return numba.types.StringLiteral

    @property
    def length(self):
        return numba.types.float64


@numba.extending.typeof_impl.register(LayoutBuilder)
def typeof_LayoutBuilder(val, c):
    if isinstance(val, Numpy):
        return NumpyType(numba.from_dtype(val._data.dtype))

    if isinstance(val, Empty):
        return EmptyType()

    if isinstance(val, ListOffset):
        return ListOffsetType(numba.from_dtype(val._offsets.dtype), val._content)

    if isinstance(val, List):
        return ListType(numba.from_dtype(val._starts.dtype), val._content)

    if isinstance(val, Regular):
        return RegularType(val._content, val._size)

    if isinstance(val, IndexedOption):
        return IndexedOptionType(numba.from_dtype(val._index.dtype), val._content)

    if isinstance(val, ByteMasked):
        return ByteMaskedType(val._content)

    if isinstance(val, BitMasked):
        return BitMaskedType(valid_when, lsb_order, val._content)

    if isinstance(val, Unmasked):
        return UnmaskedType(val._content)

    if isinstance(val, Record):
        return RecordType(val._content, val._fields)

    if isinstance(val, Tuple):
        return TupleType(val._content)

    if isinstance(val, EmptyRecord):
        return EmptyRecord(is_tuple)

    if isinstance(val, Union):
        return UnionType(numba.from_dtype(val._index.dtype), val._content)

    return LayoutBuilderType(val)


@numba.extending.overload(len)
def LayoutBuilderType_len(builder):
    if (
        isinstance(builder, NumpyType)
        or isinstance(builder, EmptyType)
        or isinstance(builder, ListOffsetType)
        or isinstance(builder, List)
        or isinstance(builder, Regular)
        or isinstance(builder, IndexedOption)
        or isinstance(builder, ByteMasked)
        or isinstance(builder, BitMasked)
        or isinstance(builder, Unmasked)
        or isinstance(builder, Record)
        or isinstance(builder, Tuple)
        or isinstance(builder, EmptyRecord)
        or isinstance(builder, Union)
    ):

        def len_impl(builder):
            return builder._length_get()

        return len_impl


########## Numpy ############################################################


@final
class Numpy(LayoutBuilder):
    def __init__(self, dtype, *, parameters=None, initial=1024, resize=8.0):
        self._data = GrowableBuffer(dtype=dtype, initial=initial, resize=resize)
        self._parameters = parameters

    @classmethod
    def _from_buffer(cls, data):
        out = cls.__new__(cls)
        out._data = data
        out._parameters = None
        return out

    def __repr__(self):
        return f"<Numpy of {self._data.dtype!r} with {self._length} items>"

    def _type(self, typestrs):
        return f"ak.numba.lb.Numpy({self._data.dtype})"

    def numbatype(self):
        return NumpyType(numba.from_dtype(self.dtype))

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def _length(self):
        return len(self._data)

    def __len__(self):
        return self._length

    def append(self, x):
        self._data.append(x)

    def extend(self, data):
        self._data.extend(data)

    def parameters(self):
        return self._parameters

    def clear(self):
        self._data.clear()

    def is_valid(self, error: str):
        return True

    def snapshot(self) -> ArrayLike:
        """
        Converts the currently accumulated data into an #ak.Array.
        """
        return ak.Array(
            ak.contents.NumpyArray(self._data.snapshot(), parameters=self._parameters)
        )


class NumpyType(numba.types.Type):
    def __init__(self, dtype):
        super().__init__(name=f"ak.numba.lb.Numpy({dtype})")
        self._dtype = dtype

    @classmethod
    def type(cls, content):
        return NumpyType(content.dtype)

    # @property
    # def dtype(self):
    #     return self._dtype

    @property
    def parameters(self):
        return numba.types.StringLiteral

    @property
    def data(self):
        return ak.numba.GrowableBufferType(self._dtype)

    @property
    def length(self):
        return numba.types.float64


@numba.extending.register_model(NumpyType)
class NumpyModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", fe_type.data),
        ]
        super().__init__(dmm, fe_type, members)


for member in ("data",):
    numba.extending.make_attribute_wrapper(NumpyType, member, "_" + member)


@numba.extending.unbox(NumpyType)
def NumpyType_unbox(typ, obj, c):
    # get PyObjects
    data_obj = c.pyapi.object_getattr_string(obj, "_data")

    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out.data = c.pyapi.to_native_value(typ.data, data_obj).value

    # decref PyObjects
    c.pyapi.decref(data_obj)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(NumpyType)
def NumpyType_box(typ, val, c):
    # get PyObject of the Numpy class and _from_buffer constructor
    Numpy_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Numpy))
    from_buffer_obj = c.pyapi.object_getattr_string(Numpy_obj, "_from_buffer")

    builder = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    data_obj = c.pyapi.from_native_value(typ.data, builder.data, c.env_manager)

    out = c.pyapi.call_function_objargs(
        from_buffer_obj,
        (data_obj,),
    )

    # decref PyObjects
    c.pyapi.decref(Numpy_obj)
    c.pyapi.decref(from_buffer_obj)

    c.pyapi.decref(data_obj)

    return out


def _from_buffer():
    ...


@numba.extending.type_callable(_from_buffer)
def Numpy_from_buffer_typer(context):
    def typer(buffer):
        if isinstance(buffer, GrowableBufferType):
            return NumpyType(buffer.dtype)

    return typer


@numba.extending.lower_builtin(_from_buffer, GrowableBufferType)
def Numpy_from_buffer_impl(context, builder, sig, args):
    out = numba.core.cgutils.create_struct_proxy(sig.return_type)(context, builder)
    out.data = args[0]

    if context.enable_nrt:
        context.nrt.incref(builder, sig.args[0], args[0])

    return out._getvalue()


@numba.extending.overload(Numpy)
def Numpy_ctor(dtype, parameters=None, initial=1024, resize=8.0):
    if isinstance(dtype, numba.types.StringLiteral):
        dt = np.dtype(dtype.literal_value)

    elif isinstance(dtype, numba.types.DTypeSpec):
        dt = numba.core.typing.npydecl.parse_dtype(dtype)

    else:
        return

    def ctor_impl(dtype, parameters=None, initial=1024, resize=8.0):
        panels = numba.typed.List([np.empty((initial,), dt)])
        length_pos = np.zeros((2,), dtype=np.int64)
        data = ak._connect.numba.growablebuffer._from_data(  # noqa: RUF100, E1111
            panels, length_pos, resize
        )

        return _from_buffer(data)

    return ctor_impl


@numba.extending.overload_method(NumpyType, "_length_get", inline="always")
def Numpy_length(builder):
    def getter(builder):
        return builder.data._length_pos[0]

    return getter


@numba.extending.overload_attribute(NumpyType, "data", inline="always")
def Numpy_buffer(builder):
    def get(builder):
        return builder._data

    return get


@numba.extending.overload_method(NumpyType, "append")
def Numpy_append(builder, datum):
    if isinstance(builder, NumpyType):

        def append(builder, datum):
            builder.data.append(datum)

        return append


@numba.extending.overload_method(NumpyType, "extend")
def Numpy_extend(builder, data):
    def extend(builder, data):
        builder.data.extend(data)

    return extend


@numba.extending.overload_method(NumpyType, "snapshot")
def Numpy_snapshot(builder):
    def snapshot(builder):
        return builder.data.snapshot()

    return snapshot


########## Empty ############################################################


@final
class Empty(LayoutBuilder):
    def __init__(self, *, parameters=None):
        self._parameters = parameters

    def __repr__(self):
        return f"<Empty with {self.length} items>"

    def _type(self, typestrs):
        return "ak.numba.lb.Empty()"

    def numbatype(self):
        return EmptyType()

    @property
    def _length(self):
        return 0

    def length(self):
        return self._length

    def __len__(self):
        return self._length

    def parameters(self):
        return self._parameters

    def clear(self):
        pass

    def is_valid(self, error: str):
        return True

    def snapshot(self) -> ArrayLike:
        return ak.Array(ak.contents.EmptyArray(parameters=self._parameters))


class EmptyType(numba.types.Type):
    def __init__(self):
        super().__init__(name="ak.numba.lb.Empty()")

    @classmethod
    def type(cls, content):
        return EmptyType()

    @property
    def length(self):
        return numba.types.int64


@numba.extending.register_model(EmptyType)
class EmptyModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("length", fe_type.length)]
        super().__init__(dmm, fe_type, members)


for member in ("length",):
    numba.extending.make_attribute_wrapper(EmptyType, member, "_" + member)


@numba.extending.unbox(EmptyType)
def EmptyType_unbox(typ, obj, c):
    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(EmptyType)
def EmptyType_box(typ, val, c):
    # get PyObject of the Empty class
    Empty_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Empty))

    out = c.pyapi.call_function_objargs(Empty_obj, ())

    # decref PyObjects
    c.pyapi.decref(Empty_obj)

    return out


@numba.extending.overload_method(EmptyType, "_length_get", inline="always")
def Empty_length(builder):
    def getter(builder):
        return builder._length

    return getter


@numba.extending.overload_method(EmptyType, "snapshot")
def Empty_snapshot(builder):
    def snapshot(builder):
        out = np.empty(0)
        return out

    return snapshot


########## ListOffset #########################################################


@final
class ListOffset(LayoutBuilder):
    def __init__(self, dtype, content, *, parameters=None, initial=1024, resize=8.0):
        self._offsets = GrowableBuffer(dtype=dtype, initial=initial, resize=resize)
        self._offsets.append(0)
        self._content = content
        self._parameters = parameters

    @classmethod
    def _from_offsets(cls, offsets, content):
        out = cls.__new__(cls)
        out._offsets = offsets
        out._content = content
        out._parameters = None
        return out

    def __repr__(self):
        return f"<ListOffset of {self._content!r} with {self._length} items>"

    def _type(self, typestrs):
        return f"ak.numba.lb.ListOffset({self._offsets.dtype}, {self._content.type})"

    def numbatype(self):
        # content_type = numba.deferred_type()
        # content_type.define(self.content.numbatype())
        return ListOffsetType(
            numba.from_dtype(self.offsets.dtype), self.content.numbatype()
        )

    @property
    def offsets(self):
        return self._offsets

    @property
    def content(self):
        return self._content

    def begin_list(self):
        return self._content

    def end_list(self):
        self._offsets.append(len(self._content))

    def parameters(self):
        return self._parameters

    def clear(self):
        self._offsets.clear()
        self._offsets.append(0)
        self._content.clear()

    @property
    def _length(self):
        return self._offsets._length - 1

    def __len__(self):
        return self._length

    def is_valid(self, error: str):
        if len(self._content) != self._offsets.last():
            error = f"ListOffset node{self._id} has content length {len(self._content)} but last offset {self._offsets.last()}"
            return False
        else:
            return self._content.is_valid(error)

    def snapshot(self) -> ArrayLike:
        """
        Converts the currently accumulated data into an #ak.Array.
        """
        content = self._content.snapshot()

        return ak.Array(
            ak.contents.listoffsetarray.ListOffsetArray(
                ak.index.Index(self._offsets.snapshot()),
                content.layout,
                parameters=self._parameters,
            )
        )


class ListOffsetType(numba.types.Type):
    def __init__(self, dtype, content):
        super().__init__(name=f"ak.numba.lb.ListOffset({dtype}, {content.type})")
        self._dtype = dtype
        self._content = content

    @classmethod
    def type(cls, content):
        return ListOffsetType(content.offsets.dtype, content.content)

    @property
    def parameters(self):
        return numba.types.StringLiteral

    @property
    def offsets(self):
        return ak.numba.GrowableBufferType(self._dtype)

    @property
    def content(self):
        return tonumbatype(self._content)

    @property
    def length(self):
        return numba.types.float64


@numba.extending.register_model(ListOffsetType)
class ListOffsetModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("offsets", fe_type.offsets),
            ("content", fe_type.content),
        ]
        super().__init__(dmm, fe_type, members)


for member in (
    "offsets",
    "content",
):
    numba.extending.make_attribute_wrapper(ListOffsetType, member, "_" + member)


@numba.extending.unbox(ListOffsetType)
def ListOffsetType_unbox(typ, obj, c):
    # get PyObjects
    offsets_obj = c.pyapi.object_getattr_string(obj, "_offsets")
    content_obj = c.pyapi.object_getattr_string(obj, "_content")

    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out.offsets = c.pyapi.to_native_value(typ.offsets, offsets_obj).value
    out.content = c.pyapi.to_native_value(typ.content, content_obj).value

    # decref PyObjects
    c.pyapi.decref(offsets_obj)
    c.pyapi.decref(content_obj)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(ListOffsetType)
def ListOffsetType_box(typ, val, c):
    # get PyObject of the ListOffset class
    ListOffset_obj = c.pyapi.unserialize(c.pyapi.serialize_object(ListOffset))

    builder = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    offsets_obj = c.pyapi.from_native_value(typ.offsets, builder.offsets, c.env_manager)
    content_obj = c.pyapi.from_native_value(typ.content, builder.content, c.env_manager)

    out = c.pyapi.call_function_objargs(
        ListOffset_obj,
        (
            offsets_obj,
            content_obj,
        ),
    )

    # decref PyObjects
    c.pyapi.decref(ListOffset_obj)

    c.pyapi.decref(offsets_obj)
    c.pyapi.decref(content_obj)

    return out


@numba.extending.overload_method(ListOffsetType, "_length_get", inline="always")
def ListOffset_length(builder):
    def getter(builder):
        return builder._offsets._length_pos[0] - 1

    return getter


@numba.extending.overload_method(ListOffsetType, "_offsets", inline="always")
def ListOffset_offsets(builder):
    def getter(builder):
        return builder._offsets

    return getter


@numba.extending.overload_method(ListOffsetType, "begin_list", inline="always")
def ListOffset_begin_list(builder):
    if isinstance(builder, ListOffsetType):

        def getter(builder):
            return builder._content

        return getter


@numba.extending.overload_method(ListOffsetType, "append")
def ListOffset_append(builder, datum):
    if isinstance(builder, ListOffsetType):

        def append(builder, datum):
            builder.append(datum)

        return append


@numba.extending.overload_method(ListOffsetType, "end_list", inline="always")
def ListOffset_end_list(builder):
    if isinstance(builder, ListOffsetType):

        def impl(builder):
            builder._offsets.append(len(builder._content))

        return impl


@numba.extending.overload_method(ListOffsetType, "extend")
def ListOffset_extend(builder, datum):
    if isinstance(builder, ListOffsetType):

        def extend(builder, datum):
            builder.extend(datum)

        return extend


@numba.extending.overload_method(ListOffsetType, "snapshot")
def ListOffset_snapshot(builder):
    def snapshot(builder):
        return builder.snapshot()

    return snapshot


########## List ############################################################


@final
class List(LayoutBuilder):
    def __init__(self, dtype, content, *, parameters=None, initial=1024, resize=8.0):
        self._starts = GrowableBuffer(dtype=dtype, initial=initial, resize=resize)
        self._stops = GrowableBuffer(dtype=dtype, initial=initial, resize=resize)
        self._content = content
        self._parameters = parameters

    def numbatype(self):
        return ListType(numba.from_dtype(self.starts.dtype), self.content.numbatype())

    @property
    def starts(self):
        return self._starts

    @property
    def stops(self):
        return self._stops

    @property
    def content(self):
        return self._content

    def begin_list(self):
        self._starts.append(len(self._content))
        return self._content

    def end_list(self):
        self._stops.append(len(self._content))

    def parameters(self):
        return self._parameters

    def clear(self):
        self._starts.clear()
        self._stops.clear()
        self._content.clear()

    @property
    def _length(self):
        return len(self._starts)

    def __len__(self):
        return self._length

    def is_valid(self, error: str):
        if len(self._starts) != len(self._stops):
            error = f"List node{self._id} has starts length {len(self._starts)} but stops length {len(self._stops)}"
        elif len(self._stops) > 0 and len(self._content) != self._stops.last():
            error = f"List node{self._id} has content length {len(self._content)} but last stops {self._stops.last()}"
            return False
        else:
            return self._content.is_valid(error)

    def snapshot(self) -> ArrayLike:
        """
        Converts the currently accumulated data into an #ak.Array.
        """
        return ak.Array(
            ak.contents.ListArray(
                ak.index.Index(self._starts.snapshot()),
                ak.index.Index(self._stops.snapshot()),
                self._content.snapshot().layout,
                parameters=self._parameters,
            )
        )


class ListType(numba.types.Type):
    def __init__(self, dtype, content):
        super().__init__(name=f"ak.numba.lb.List({dtype}, {content.type})")
        self._dtype = dtype
        self._content = content

    @classmethod
    def type(cls, content):
        return ListType(content.starts.dtype, content.content)

    @property
    def parameters(self):
        return numba.types.StringLiteral

    @property
    def starts(self):
        return ak.numba.GrowableBufferType(self._dtype)

    @property
    def stops(self):
        return ak.numba.GrowableBufferType(self._dtype)

    @property
    def content(self):
        return tonumbatype(self._content)

    @property
    def length(self):
        return numba.types.float64


@numba.extending.register_model(ListType)
class ListModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("starts", fe_type.starts),
            ("stops", fe_type.starts),
            ("content", fe_type.content),
        ]
        super().__init__(dmm, fe_type, members)


for member in (
    "starts",
    "stops",
    "content",
):
    numba.extending.make_attribute_wrapper(ListType, member, "_" + member)


@numba.extending.unbox(ListType)
def ListType_unbox(typ, obj, c):
    # get PyObjects
    starts_obj = c.pyapi.object_getattr_string(obj, "_starts")
    stops_obj = c.pyapi.object_getattr_string(obj, "_stops")
    content_obj = c.pyapi.object_getattr_string(obj, "_content")

    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out.starts = c.pyapi.to_native_value(typ.starts, starts_obj).value
    out.stops = c.pyapi.to_native_value(typ.stops, starts_obj).value
    out.content = c.pyapi.to_native_value(typ.content, content_obj).value

    # decref PyObjects
    c.pyapi.decref(starts_obj)
    c.pyapi.decref(stops_obj)
    c.pyapi.decref(content_obj)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(ListType)
def ListType_box(typ, val, c):
    # get PyObject of the List class
    List_obj = c.pyapi.unserialize(c.pyapi.serialize_object(List))

    builder = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    starts_obj = c.pyapi.from_native_value(typ.starts, builder.starts, c.env_manager)
    stops_obj = c.pyapi.from_native_value(typ.stops, builder.stops, c.env_manager)
    content_obj = c.pyapi.from_native_value(typ.content, builder.content, c.env_manager)

    out = c.pyapi.call_function_objargs(
        List_obj,
        (
            starts_obj,
            content_obj,
        ),
    )

    # decref PyObjects
    c.pyapi.decref(List_obj)

    c.pyapi.decref(starts_obj)
    c.pyapi.decref(content_obj)

    return out


@numba.extending.overload_method(ListType, "_length_get", inline="always")
def List_length(builder):
    def getter(builder):
        return builder._starts._length_pos[0] - 1

    return getter


@numba.extending.overload_method(ListType, "_offsets", inline="always")
def List_offsets(builder):
    def getter(builder):
        return builder._starts

    return getter


@numba.extending.overload_method(ListType, "begin_list", inline="always")
def List_begin_list(builder):
    if isinstance(builder, ListType):

        def getter(builder):
            builder._starts.append(len(builder._content))
            return builder._content

        return getter


@numba.extending.overload_method(ListType, "append")
def List_append(builder, datum):
    if isinstance(builder, ListType):

        def append(builder, datum):
            builder.append(datum)

        return append


@numba.extending.overload_method(ListType, "end_list", inline="always")
def List_end_list(builder):
    if isinstance(builder, ListType):

        def impl(builder):
            builder._stops.append(len(builder._content))

        return impl


@numba.extending.overload_method(ListType, "extend")
def List_extend(builder, datum):
    if isinstance(builder, ListType):

        def extend(builder, datum):
            builder.extend(datum)

        return extend


@numba.extending.overload_method(ListType, "snapshot")
def List_snapshot(builder):
    def snapshot(builder):
        return builder.snapshot()

    return snapshot


########## Regular ############################################################


@final
class Regular(LayoutBuilder):
    def __init__(self, content, size, *, parameters=None):
        self._length = 0
        self._content = content
        self._size = size
        self._parameters = parameters

    @property
    def content(self):
        return self._content

    def begin_list(self):
        return self._content

    def end_list(self):
        self._length += 1

    def parameters(self):
        return self._parameters

    def clear(self):
        self._content.clear()

    def length(self):
        return self._length

    def __len__(self):
        return self._length

    def is_valid(self, error: str):
        if len(self._content) != self._length * self._size:
            error = f"Regular node{self._id} has content length {len(self._content)}, but length {self._length} and size {self._size}"
            return False
        else:
            return self._content.is_valid(error)

    def snapshot(self) -> ArrayLike:
        """
        Converts the currently accumulated data into an #ak.Array.
        """
        return ak.Array(
            ak.contents.RegularArray(
                self._content.snapshot().layout,
                self._size,
                self.length(),
                parameters=self._parameters,
            )
        )


########## Indexed ############################################################


@final
class Indexed(LayoutBuilder):
    def __init__(self, dtype, content, *, parameters=None):
        self._last_valid = -1
        self._index = GrowableBuffer(dtype=dtype)
        self._content = content
        self._parameters = parameters

    @property
    def content(self):
        return self._content

    def append(self, datum):
        self._last_valid = len(self._content)
        self._index.append(self._last_valid)
        self._content.append(datum)

    def append_index(self):
        self._last_valid = len(self._content)
        self._index.append(self._last_valid)
        return self._content

    def extend(self, data):
        start = len(self._content)
        stop = start + len(data)
        self._last_valid = stop - 1
        self._index.extend(list(range(start, stop)))
        self._content.extend(data)

    def extend_index(self, size):
        start = len(self._content)
        stop = start + size
        self._last_valid = stop - 1
        self._index.extend(list(range(start, stop)))
        return self._content

    def parameters(self):
        return self._parameters

    def clear(self):
        self._last_valid = -1
        self._index.clear()
        self._content.clear()

    def length(self):
        return self._index._length

    def __len__(self):
        return self.length()

    def is_valid(self, error: str):
        if len(self._content) != self._index.length():
            error = f"Indexed node{self._id} has content length {len(self._content)} but index length {self._index.length()}"
            return False
        elif len(self._content) != self._last_valid + 1:
            error = f"Indexed node{self._id} has content length {len(self._content)} but last valid index is {self._last_valid}"
        else:
            return self._content.is_valid(error)

    def snapshot(self) -> ArrayLike:
        """
        Converts the currently accumulated data into an #ak.Array.
        """
        return ak.Array(
            ak.contents.IndexedArray(
                ak.index.Index64(self._index.snapshot()),
                self._content.snapshot().layout,
                parameters=self._parameters,
            )
        )


########## IndexedOption #######################################################


@final
class IndexedOption(LayoutBuilder):
    def __init__(self, dtype, content, *, parameters=None):
        self._last_valid = -1
        self._index = GrowableBuffer(dtype=dtype)
        self._content = content
        self._parameters = parameters

    @property
    def content(self):
        return self._content

    def append(self, datum):
        self._last_valid = len(self._content)
        self._index.append(self._last_valid)
        self._content.append(datum)

    def append_index(self):
        self._last_valid = len(self._content)
        self._index.append(self._last_valid)
        return self._content

    def extend(self, data):
        start = len(self._content)
        stop = start + len(data)
        self._last_valid = stop - 1
        self._index.extend(list(range(start, stop)))
        self._content.extend(data)

    def extend_index(self, size):
        start = len(self._content)
        stop = start + size
        self._last_valid = stop - 1
        self._index.extend(list(range(start, stop)))
        return self._content

    def append_null(self):
        self._index.append(-1)

    def extend_null(self, size):
        self._index.extend([-1] * size)

    def parameters(self):
        return self._parameters

    def clear(self):
        self._last_valid = -1
        self._index.clear()
        self._content.clear()

    def length(self):
        return self._index._length

    def __len__(self):
        return self.length()

    def is_valid(self, error: str):
        if len(self._content) != self._last_valid + 1:
            error = f"Indexed node{self._id} has content length {len(self._content)} but last valid index is {self._last_valid}"
            return False
        else:
            return self._content.is_valid(error)

    def snapshot(self) -> ArrayLike:
        """
        Converts the currently accumulated data into an #ak.Array.
        """
        return ak.Array(
            ak.contents.IndexedOptionArray(
                ak.index.Index64(self._index.snapshot()),
                self._content.snapshot().layout,
                parameters=self._parameters,
            )
        )


########## ByteMasked #########################################################


@final
class ByteMasked(LayoutBuilder):
    def __init__(
        self,
        content,
        *,
        parameters=None,
        valid_when=True,
    ):
        self._mask = GrowableBuffer("bool")
        self._content = content
        self._valid_when = valid_when
        self._parameters = parameters

    @property
    def content(self):
        return self._content

    def valid_when(self):
        return self._valid_when

    def append_valid(self):
        self._mask.append(self._valid_when)
        return self._content

    def extend_valid(self, size):
        self._mask.extend([self._valid_when] * size)
        return self._content

    def append_null(self):
        self._mask.append(not self._valid_when)
        return self._content

    def extend_null(self, size):
        self._mask.extend([not self._valid_when] * size)
        return self._content

    def parameters(self):
        return self._parameters

    def clear(self):
        self._mask.clear()
        self._content.clear()

    def length(self):
        return len(self._mask)

    def __len__(self):
        return self.length()

    def is_valid(self, error: str):
        if len(self._content) != self._mask.length():
            error = f"ByteMasked node{self._id} has content length {len(self._content)} but mask length {len(self._stops)}"
            return False
        else:
            return self._content.is_valid(error)

    def snapshot(self) -> ArrayLike:
        """
        Converts the currently accumulated data into an #ak.Array.
        """
        return ak.Array(
            ak.contents.ByteMaskedArray(
                ak.index.Index8(self._mask.snapshot()),
                self._content.snapshot().layout,
                valid_when=self._valid_when,
                parameters=self._parameters,
            )
        )


########## BitMasked #########################################################


@final
class BitMasked(LayoutBuilder):
    def __init__(
        self,
        valid_when,
        lsb_order,
        content,
        *,
        parameters=None,
    ):
        self._mask = GrowableBuffer("uint8")
        self._content = content
        self._valid_when = valid_when
        self._lsb_order = lsb_order
        self._current_byte = np.uint8(0)
        self._mask.append(self._current_byte)
        self._current_index = 0
        if self._lsb_order:
            self._cast = np.array(
                [
                    np.uint8(1 << 0),
                    np.uint8(1 << 1),
                    np.uint8(1 << 2),
                    np.uint8(1 << 3),
                    np.uint8(1 << 4),
                    np.uint8(1 << 5),
                    np.uint8(1 << 6),
                    np.uint8(1 << 7),
                ]
            )
        else:
            self._cast = np.array(
                [
                    np.uint8(128 >> 0),
                    np.uint8(128 >> 1),
                    np.uint8(128 >> 2),
                    np.uint8(128 >> 3),
                    np.uint8(128 >> 4),
                    np.uint8(128 >> 5),
                    np.uint8(128 >> 6),
                    np.uint8(128 >> 7),
                ]
            )
        self._parameters = parameters

    @property
    def content(self):
        return self._content

    def valid_when(self):
        return self._valid_when

    def lsb_order(self):
        return self._lsb_order

    def _append_begin(self):
        """
        Private helper function.
        """
        if self._current_index == 8:
            self._current_byte = np.uint8(0)
            self._mask.append(self._current_byte)
            self._current_index = 0

    def _append_end(self):
        """
        Private helper function.
        """
        self._current_index += 1
        if self._valid_when:
            # 0 indicates null, 1 indicates valid
            self._mask._panels[-1][self._mask._pos - 1] = self._current_byte
        else:
            # 0 indicates valid, 1 indicates null
            self._mask._panels[-1][self._mask._pos - 1] = ~self._current_byte

    def append_valid(self):
        self._append_begin()
        # current_byte_ and cast_: 0 indicates null, 1 indicates valid
        self._current_byte |= self._cast[self._current_index]
        self._append_end()
        return self._content

    def extend_valid(self, size):
        # Just an interface; not actually faster than calling append many times.
        for _ in range(size):
            self.append_valid()
        return self._content

    def append_null(self):
        self._append_begin()
        # current_byte_ and cast_ default to null, no change
        self._append_end()
        return self._content

    def extend_null(self, size):
        # Just an interface; not actually faster than calling append many times.
        for _ in range(size):
            self.append_null()
        return self._content

    def parameters(self):
        return self._parameters

    def clear(self):
        self._mask.clear()
        self._content.clear()

    def length(self):
        return (len(self._mask) - 1) * 8 + self._current_index

    def __len__(self):
        return self.length()

    def is_valid(self, error: str):
        if len(self._content) != self.length():
            error = f"BitMasked node{self._id} has content length {len(self._content)} but bit mask length {self.length()}"
            return False
        else:
            return self._content.is_valid(error)

    def snapshot(self) -> ArrayLike:
        """
        Converts the currently accumulated data into an #ak.Array.
        """
        return ak.Array(
            ak.contents.BitMaskedArray(
                ak.index.Index(self._mask.snapshot()),
                self._content.snapshot().layout,
                valid_when=self._valid_when,
                length=self.length(),
                lsb_order=self._lsb_order,
                parameters=self._parameters,
            )
        )


########## Unmasked #########################################################


@final
class Unmasked(LayoutBuilder):
    def __init__(self, content, *, parameters=None):
        self._content = content
        self._parameters = parameters

    @property
    def content(self):
        return self._content

    def append_valid(self):
        return self._content

    def extend_valid(self, size):
        return self._content

    def parameters(self):
        return self._parameters

    def clear(self):
        self._content.clear()

    def length(self):
        return len(self._content)

    def __len__(self):
        return self.length()

    def is_valid(self, error: str):
        return self._content.is_valid(error)

    def snapshot(self) -> ArrayLike:
        """
        Converts the currently accumulated data into an #ak.Array.
        """
        return ak.Array(
            ak.contents.UnmaskedArray(
                self._content.snapshot().layout,
                parameters=self._parameters,
            )
        )


########## Record #########################################################


@final
class Record(LayoutBuilder):
    def __init__(self, contents, fields, *, parameters=None):
        assert len(fields) != 0
        self._contents = contents
        self._fields = fields
        self._first_content = self._contents[0]
        self._parameters = parameters

    def field(self, name):
        return self._contents[self._fields.index(name)]

    def parameters(self):
        return self._parameters

    def clear(self):
        for pair in self._field_pairs.values():
            pair.content.clear()

    def length(self):
        return len(self._first_content)

    def __len__(self):
        return self.length()

    def is_valid(self, error: str):
        length = -1
        # for pair in self._field_pairs.values():
        for i in enumerate(self._contents):
            if length == -1:
                length = len(self._contents[i])
            elif length != len(self._contents[i]):
                error = f"Record node{self._id} has field {self._fields[i]} length {len(self._contents[i])} that differs from the first length {length}"
                return False
        for content in self._contents:
            if not content.is_valid(error):
                return False
        return True

    def snapshot(self) -> ArrayLike:
        """
        Converts the currently accumulated data into an #ak.Array.
        """
        contents = []
        for content in self._contents:
            contents.append(content.snapshot().layout)

        return ak.Array(
            ak.contents.RecordArray(
                contents,
                self._fields,
                parameters=self._parameters,
            )
        )


########## Tuple #######################################################


@final
class Tuple(LayoutBuilder):
    def __init__(self, contents, *, parameters=None):
        assert len(contents) != 0
        self._contents = contents
        self._first_content = contents[0]
        self._parameters = parameters

    def index(self, at):
        return self._contents[at]

    def parameters(self):
        return self._parameters

    def clear(self):
        for _content in self._contents:
            _content.clear()

    def length(self):
        return len(self._first_content)

    def __len__(self):
        return self.length()

    def is_valid(self, error: str):
        length = -1
        for index, content in enumerate(self._contents):
            if length == -1:
                length = len(content)
            elif length != len(content):
                error = f"Tuple node{self._id} has index {index} length {len(content)} that differs from the first length {length}"
                return False
        for content in self._contents:
            if not content.is_valid(error):
                return False
        return True

    def snapshot(self) -> ArrayLike:
        """
        Converts the currently accumulated data into an #ak.Array.
        """
        contents = []
        for content in self._contents:
            contents.append(content.snapshot().layout)

        return ak.Array(
            ak.contents.RecordArray(
                contents,
                None,
                parameters=self._parameters,
            )
        )


########## EmptyRecord #######################################################


@final
class EmptyRecord(LayoutBuilder):
    def __init__(self, is_tuple, *, parameters=None):
        self._length = 0
        self._is_tuple = is_tuple
        self._parameters = parameters

    def append(self):
        self._length += 1

    def extend(self, size):
        self._length += size

    def parameters(self):
        return self._parameters

    def clear(self):
        self._length = 0

    def length(self):
        return self._length

    def __len__(self):
        return self.length()

    def is_valid(self, error: str):
        return True

    def snapshot(self) -> ArrayLike:
        """
        Converts the currently accumulated data into an #ak.Array.
        """
        contents = [] if self._is_tuple else {}

        return ak.Array(
            ak.contents.RecordArray(
                contents,
                None,
                self.length(),
                parameters=self._parameters,
            )
        )


########## Union #######################################################


@final
class Union(LayoutBuilder):
    def __init__(self, dtype, contents, *, parameters=None):
        self._last_valid_index = [-1] * len(contents)
        self._tags = GrowableBuffer("int8")
        self._index = GrowableBuffer(dtype=dtype)
        self._contents = contents
        self._parameters = parameters

    def append_index(self, tag):
        which_content = self._contents[tag]
        next_index = len(which_content)
        self._last_valid_index[tag] = next_index
        self._tags.append(tag)
        self._index.append(next_index)
        return which_content

    def parameters(self):
        return self._parameters

    def clear(self):
        for tag, _value in self._last_valid_index:
            self._last_valid_index[tag] = -1
        self._tags.clear()
        self._index.clear()
        for content in self._contents:
            content.clear()

    def length(self):
        return len(self._tags)

    def __len__(self):
        return self.length()

    def is_valid(self, error: str):
        for tag, _value in self._last_valid_index:
            if self._contents[tag].length() != self._last_valid_index[tag] + 1:
                error = f"Union node{self._id} has content {tag} length {self._contents[tag].length()} but last valid index is {self._last_valid_index[tag]}"
                return False
        for content in self._contents:
            if not content.is_valid(error):
                return False
        return True

    def snapshot(self) -> ArrayLike:
        """
        Converts the currently accumulated data into an #ak.Array.
        """
        contents = []
        for content in self._contents:
            contents.append(content.snapshot().layout)

        return ak.Array(
            ak.contents.UnionArray(
                ak.index.Index8(self._tags.snapshot()),
                ak.index.Index64(self._index.snapshot()),
                contents,
                parameters=self._parameters,
            )
        )
