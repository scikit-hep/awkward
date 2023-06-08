# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import math

import numba
import numba.core.typing.npydecl
import numpy as np

import awkward as ak
from awkward._connect.numba.growablebuffer import GrowableBuffer, GrowableBufferType
from awkward._nplikes.numpylike import ArrayLike
from awkward._typing import final


class LayoutBuilder:
    def __init__(self, builder, *, parameters=None):
        self._builder = builder
        self._parameters = parameters

    @property
    def builder(self):
        return self._builder

    def clear(self):
        return self._builder.clear()

    def is_valid(self, error: str):
        return self._builder.is_valid(error)

    def _type(self, typestrs):
        raise NotImplementedError


def to_numbatype(builder):
    if isinstance(builder, Numpy):
        return Numpy.numbatype(builder)
    elif isinstance(builder, Empty):
        return Empty.numbatype(builder)
    elif isinstance(builder, ListOffset):
        return ListOffset.numbatype(builder)
    elif isinstance(builder, Regular):
        return Regular.numbatype(builder)
    elif isinstance(builder, IndexedOption):
        return IndexedOption.numbatype(builder)
    elif isinstance(builder, ByteMasked):
        return ByteMasked.numbatype(builder)
    elif isinstance(builder, BitMasked):
        return BitMasked.numbatype(builder)
    elif isinstance(builder, Unmasked):
        return Unmasked.numbatype(builder)
    elif isinstance(builder, Record):
        return Record.numbatype(builder)
    elif isinstance(builder, Tuple):
        return Tuple.numbatype(builder)
    elif isinstance(builder, Union):
        return Union.numbatype(builder)
    elif isinstance(builder, tuple):
        return numba.types.Tuple([to_numbatype(it) for it in builder])
    else:
        return builder


class LayoutBuilderType(numba.types.Type):
    def __init__(self, parameters):
        super().__init__(name="ak.numba.lb.LayoutBuilder()")

    @property
    def parameters(self):
        return numba.types.StringLiteral(self._parameters)

    @property
    def length(self):
        return numba.types.int64


@numba.extending.typeof_impl.register(LayoutBuilder)
def typeof_LayoutBuilder(val, c):
    if isinstance(val, BitMasked):
        return BitMaskedType(
            numba.from_dtype(val._mask.dtype),
            val._content,
            val._valid_when,
            val._lsb_order,
            val._parameters,
        )

    elif isinstance(val, ByteMasked):
        return ByteMaskedType(
            numba.from_dtype(val._mask.dtype),
            val._content,
            val._valid_when,
            val._parameters,
        )

    elif isinstance(val, Empty):
        return EmptyType(val._parameters)

    elif isinstance(val, IndexedOption):
        return IndexedOptionType(
            numba.from_dtype(val._index.dtype), val._content, val._parameters
        )

    elif isinstance(val, ListOffset):
        return ListOffsetType(
            numba.from_dtype(val._offsets.dtype), val._content, val._parameters
        )

    elif isinstance(val, Numpy):
        return NumpyType(numba.from_dtype(val._data.dtype), val._parameters)

    elif isinstance(val, Record):
        return RecordType(val._contents, val._fields, val._parameters)

    elif isinstance(val, Regular):
        return RegularType(val._content, val._size, val._parameters)

    elif isinstance(val, Tuple):
        return TupleType(val._contents)

    elif isinstance(val, Union):
        return UnionType(numba.from_dtype(val._index.dtype), val._contents)

    elif isinstance(val, Unmasked):
        return UnmaskedType(val._content, val._parameters)

    else:
        raise TypeError("unrecognized LayoutBuilder type")


@numba.extending.overload(len)
def LayoutBuilderType_len(builder):
    if isinstance(
        builder,
        (
            BitMaskedType,
            ByteMaskedType,
            EmptyType,
            IndexedOptionType,
            ListOffsetType,
            NumpyType,
            RecordType,
            RegularType,
            TupleType,
            UnionType,  # append_index (Literal int) updates tags and index -> content
            # and content (Literal int) method
            UnmaskedType,
        ),
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
        return f"ak.numba.lb.Numpy({self._data.dtype}, parameters={self._parameters})"

    def type(self):
        return f"ak.numba.lb.Numpy({self._data.dtype}, parameters={self._parameters})"

    def numbatype(self):
        return NumpyType(
            numba.from_dtype(self.dtype), numba.types.StringLiteral(self._parameters)
        )

    @property
    def _length(self):
        return len(self._data)

    def __len__(self):
        return self._length

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def parameters(self):
        return self._parameters

    def append(self, x):
        self._data.append(x)

    def extend(self, data):
        self._data.extend(data)

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
    def __init__(self, dtype, parameters):
        super().__init__(
            name=f"ak.numba.lb.Numpy({dtype}, parameters={parameters.literal_value if isinstance(parameters, numba.types.Literal) else None})"
        )
        self._dtype = dtype
        self._parameters = parameters

    @classmethod
    def type(cls):
        return NumpyType(cls.dtype, cls.parameters)

    @property
    def dtype(self):
        return self._dtype

    @property
    def parameters(self):
        return numba.types.StringLiteral(self._parameters)

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
    raise RuntimeError("_from_buffer Python function is only implemented in Numba")


@numba.extending.type_callable(_from_buffer)
def Numpy_from_buffer_typer(context):
    def typer(buffer):
        if isinstance(buffer, GrowableBufferType):
            return NumpyType(buffer.dtype, parameters=None)

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


@numba.extending.overload_attribute(NumpyType, "dtype", inline="always")
def Numpy_dtype(builder):
    def get(builder):
        return builder._data.dtype

    return get


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
        return f"ak.numba.lb.Empty(parameters={self.parameters})"

    def type(self):
        return f"ak.numba.lb.Empty(parameters={self.parameters})"

    def numbatype(self):
        return EmptyType(numba.types.StringLiteral(self._parameters))

    @property
    def _length(self):
        return 0

    def __len__(self):
        return self._length

    @property
    def parameters(self):
        return self._parameters

    def clear(self):
        pass

    def is_valid(self, error: str):
        return True

    def snapshot(self) -> ArrayLike:
        return ak.Array(ak.contents.EmptyArray(parameters=self._parameters))


class EmptyType(numba.types.Type):
    def __init__(self, parameters):
        super().__init__(
            name=f"ak.numba.lb.Empty(parameters={parameters.literal_value if isinstance(parameters, numba.types.Literal) else None})"
        )
        self._parameters = parameters

    @classmethod
    def type(cls):
        return EmptyType()

    @property
    def parameters(self):
        return numba.types.StringLiteral(self._parameters)

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

    def __repr__(self):
        return f"ak.numba.lb.ListOffset({self._offsets.dtype}, {self._content}, parameters={self._parameters})"

    def type(self):
        return f"ak.numba.lb.ListOffset({self._offsets.dtype}, {self._content}, parameters={self._parameters})"

    def numbatype(self):
        return ListOffsetType(
            numba.from_dtype(self.offsets.dtype),
            self.content,
            numba.types.StringLiteral(self._parameters),
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
        return self._offsets._length_pos[0] - 1

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
    def __init__(self, dtype, content, parameters):
        super().__init__(
            name=f"ak.numba.lb.ListOffset({dtype}, {content.type()}, parameters={parameters.literal_value if isinstance(parameters, numba.types.Literal) else None})"
        )
        self._dtype = dtype
        self._content = content
        self._parameters = parameters

    @classmethod
    def type(cls):
        return ListOffsetType(cls.offsets.dtype, cls.content, cls.parameters)

    @property
    def parameters(self):
        return numba.types.StringLiteral(self._parameters)

    @property
    def offsets(self):
        return ak.numba.GrowableBufferType(self._dtype)

    @property
    def content(self):
        return to_numbatype(self._content)

    @property
    def length(self):
        return numba.types.int64


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


@numba.extending.overload_method(ListOffsetType, "end_list", inline="always")
def ListOffset_end_list(builder):
    if isinstance(builder, ListOffsetType):

        def impl(builder):
            builder._offsets.append(len(builder._content))

        return impl


########## Regular ############################################################


@final
class Regular(LayoutBuilder):
    def __init__(self, content, size, *, parameters=None):
        self._content = content
        self._size = size
        self._parameters = parameters

        if size < 1:
            raise ValueError("unsupported feature: size must be at least 1")

    def __repr__(self):
        return f"ak.numba.lb.Regular({self._content}, {self._size}, parameters={self._parameters})"

    def type(self):
        return f"ak.numba.lb.Regular({self._content}, {self._size}, parameters={self._parameters})"

    def numbatype(self):
        return RegularType(
            self.content,
            self.size,
            numba.types.StringLiteral(self._parameters),
        )

    @property
    def content(self):
        return self._content

    @property
    def size(self):
        return self._size

    @property
    def _length(self):
        return math.floor(len(self.content) / self.size)

    def begin_list(self):
        return self.content

    def end_list(self):
        pass

    def parameters(self):
        return self._parameters

    def clear(self):
        self.content.clear()

    def __len__(self):
        return self._length

    def is_valid(self, error: str):  # structure_valid
        if len(self.content) != self._length * self.size:
            error = f"Regular node{self._id} has content length {len(self.content)}, but length {self._length} and size {self.size}"
            return False
        else:
            return self.content.is_valid(error)

    def snapshot(self) -> ArrayLike:
        """
        Converts the currently accumulated data into an #ak.Array.
        """
        return ak.Array(
            ak.contents.RegularArray(
                self._content.snapshot().layout,
                self._size,
                self._length,
                parameters=self._parameters,
            )
        )


class RegularType(numba.types.Type):
    def __init__(self, content, size, parameters):
        super().__init__(
            name=f"ak.numba.lb.Regular({content.type()}, {size}, parameters={parameters.literal_value if isinstance(parameters, numba.types.Literal) else None})"
        )
        self._content = content
        self._size = size
        self._parameters = parameters

    @classmethod
    def type(cls):
        return RegularType(cls.content, cls.size, cls.parameters)

    @property
    def parameters(self):
        return numba.types.StringLiteral(self._parameters)

    @property
    def content(self):
        return to_numbatype(self._content)

    @property
    def size(self):
        return numba.types.int64


@numba.extending.register_model(RegularType)
class RegularModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("content", fe_type.content),
            ("size", fe_type.size),
        ]
        super().__init__(dmm, fe_type, members)


for member in (
    "content",
    "size",
):
    numba.extending.make_attribute_wrapper(RegularType, member, "_" + member)


@numba.extending.unbox(RegularType)
def RegularType_unbox(typ, obj, c):
    # get PyObjects
    content_obj = c.pyapi.object_getattr_string(obj, "_content")
    size_obj = c.pyapi.object_getattr_string(obj, "_size")

    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out.content = c.pyapi.to_native_value(typ.content, content_obj).value
    out.size = c.pyapi.to_native_value(typ.size, size_obj).value

    # decref PyObjects
    c.pyapi.decref(content_obj)
    c.pyapi.decref(size_obj)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(RegularType)
def RegularType_box(typ, val, c):
    # get PyObject of the Regular class
    Regular_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Regular))

    builder = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    content_obj = c.pyapi.from_native_value(typ.content, builder.content, c.env_manager)
    size_obj = c.pyapi.from_native_value(typ.size, builder.size, c.env_manager)

    out = c.pyapi.call_function_objargs(
        Regular_obj,
        (
            content_obj,
            size_obj,
        ),
    )

    # decref PyObjects
    c.pyapi.decref(Regular_obj)

    c.pyapi.decref(content_obj)
    c.pyapi.decref(size_obj)

    return out


@numba.extending.overload_method(RegularType, "_length_get", inline="always")
def Regular_length(builder):
    def getter(builder):
        return math.floor(len(builder._content) / builder._size)

    return getter


@numba.extending.overload_method(RegularType, "_size", inline="always")
def Regular_starts(builder):
    def getter(builder):
        return builder._size

    return getter


@numba.extending.overload_method(RegularType, "begin_list", inline="always")
def Regular_begin_list(builder):
    if isinstance(builder, RegularType):

        def getter(builder):
            return builder._content

        return getter


@numba.extending.overload_method(RegularType, "end_list", inline="always")
def Regular_end_list(builder):
    if isinstance(builder, RegularType):

        def impl(builder):
            pass

        return impl


@numba.extending.overload_method(RegularType, "snapshot")
def Regular_snapshot(builder):
    def snapshot(builder):
        return builder.snapshot()

    return snapshot


########## IndexedOption #######################################################


@final
class IndexedOption(LayoutBuilder):
    def __init__(self, dtype, content, *, parameters=None, initial=1024, resize=8.0):
        self._last_valid = -1
        self._index = GrowableBuffer(dtype=dtype, initial=initial, resize=resize)
        self._content = content
        self._parameters = parameters

    def __repr__(self):
        return f"ak.numba.lb.IndexedOption({self._index.dtype}, {self._content}, parameters={self._parameters})"

    def type(self):
        return f"ak.numba.lb.IndexedOption({self._index.dtype}, {self._content}, parameters={self._parameters})"

    def numbatype(self):
        return IndexedOptionType(
            numba.from_dtype(self.index.dtype),
            self.content,
            numba.types.StringLiteral(self._parameters),
        )

    @property
    def index(self):
        return self._index

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

    @property
    def _length(self):
        return self._index._length

    def __len__(self):
        return self._length

    def is_valid(self, error: str):
        if len(self._content) != self._last_valid + 1:
            error = f"IndexedOption has content length {len(self._content)} but last valid index is {self._last_valid}"
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


class IndexedOptionType(numba.types.Type):
    def __init__(self, dtype, content, parameters):
        super().__init__(
            name=f"ak.numba.lb.IndexedOption({dtype}, {content.type()}, parameters={parameters.literal_value if isinstance(parameters, numba.types.Literal) else None})"
        )
        self._dtype = dtype
        self._content = content
        self._parameters = parameters

    @classmethod
    def type(cls):
        return IndexedOptionType(cls.index.dtype, cls.content, cls.parameters)

    @property
    def parameters(self):
        return numba.types.StringLiteral(self._parameters)

    @property
    def index(self):
        return ak.numba.GrowableBufferType(self._dtype)

    @property
    def content(self):
        return to_numbatype(self._content)

    @property
    def length(self):
        return numba.types.int64


@numba.extending.register_model(IndexedOptionType)
class IndexedOptionModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("index", fe_type.index),
            ("content", fe_type.content),
        ]
        super().__init__(dmm, fe_type, members)


for member in (
    "index",
    "content",
):
    numba.extending.make_attribute_wrapper(IndexedOptionType, member, "_" + member)


@numba.extending.unbox(IndexedOptionType)
def IndexedOptionType_unbox(typ, obj, c):
    # get PyObjects
    index_obj = c.pyapi.object_getattr_string(obj, "_index")
    content_obj = c.pyapi.object_getattr_string(obj, "_content")

    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out.index = c.pyapi.to_native_value(typ.index, index_obj).value
    out.content = c.pyapi.to_native_value(typ.content, content_obj).value

    # decref PyObjects
    c.pyapi.decref(index_obj)
    c.pyapi.decref(content_obj)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(IndexedOptionType)
def IndexedOptionType_box(typ, val, c):
    # get PyObject of the Indexed class
    IndexedOption_obj = c.pyapi.unserialize(c.pyapi.serialize_object(IndexedOption))

    builder = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    index_obj = c.pyapi.from_native_value(typ.index, builder.index, c.env_manager)
    content_obj = c.pyapi.from_native_value(typ.content, builder.content, c.env_manager)

    out = c.pyapi.call_function_objargs(
        IndexedOption_obj,
        (
            index_obj,
            content_obj,
        ),
    )

    # decref PyObjects
    c.pyapi.decref(IndexedOption_obj)

    c.pyapi.decref(index_obj)
    c.pyapi.decref(content_obj)

    return out


@numba.extending.overload_method(IndexedOptionType, "_length_get", inline="always")
def IndexedOption_length(builder):
    def getter(builder):
        return builder._index._length_pos[0]

    return getter


@numba.extending.overload_method(IndexedOptionType, "_index", inline="always")
def IndexedOption_index(builder):
    def getter(builder):
        return builder._index

    return getter


@numba.extending.overload_method(IndexedOptionType, "append")
def IndexedOption_append(builder, datum):
    if isinstance(builder, IndexedOptionType):

        def append(builder, datum):
            builder._index.append(len(builder._content))
            builder._content.append(datum)

        return append


@numba.extending.overload_method(IndexedOptionType, "extend")
def IndexedOption_extend(builder, data):
    def extend(builder, data):
        start = len(builder._content)
        stop = start + len(data)
        builder._index.extend(list(range(start, stop)))
        builder._content.extend(data)

    return extend


@numba.extending.overload_method(IndexedOptionType, "append_null")
def IndexedOption_append_null(builder):
    if isinstance(builder, IndexedOptionType):

        def append_null(builder):
            builder._index.append(-1)

        return append_null


@numba.extending.overload_method(IndexedOptionType, "extend_null")
def IndexedOption_extend_null(builder, size):
    def extend_null(builder, size):
        builder._index.extend([-1] * size)

    return extend_null


########## ByteMasked #########################################################


@final
class ByteMasked(LayoutBuilder):
    def __init__(
        self,
        dtype,  # mask must be "bool"
        content,
        *,
        valid_when=True,
        parameters=None,
        initial=1024,
        resize=8.0,
    ):
        self._mask = GrowableBuffer(dtype=dtype, initial=initial, resize=resize)
        self._content = content
        self._valid_when = valid_when
        self._parameters = parameters

    def __repr__(self):
        return f"ak.numba.lb.ByteMasked({self._mask.dtype}, {self._content}, valid_when={self._valid_when}, parameters={self._parameters})"

    def type(self):
        return f"ak.numba.lb.ByteMasked({self._mask.dtype}, {self._content}, valid_when={self._valid_when}, parameters={self._parameters})"

    def numbatype(self):
        return ByteMaskedType(
            numba.from_dtype(self._mask.dtype),
            self.content,
            self.valid_when,
            numba.types.StringLiteral(self._parameters),
        )

    @property
    def content(self):
        return self._content

    @property
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

    @property
    def _length(self):
        return len(self._mask)

    def __len__(self):
        return self._length

    def is_valid(self, error: str):
        if len(self._content) != len(self._mask):
            error = f"ByteMasked has content length {len(self._content)} but mask length {len(self._mask)}"
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


class ByteMaskedType(numba.types.Type):
    def __init__(self, dtype, content, valid_when, parameters):
        super().__init__(
            name=f"ak.numba.lb.ByteMasked({dtype}, {content.type()}, valid_when={valid_when}, parameters={parameters.literal_value if isinstance(parameters, numba.types.Literal) else None})"
        )
        self._dtype = dtype
        self._content = content
        self._valid_when = valid_when
        self._parameters = parameters

    @classmethod
    def type(cls):
        return ByteMaskedType(cls.content, cls.valid_when, cls.parameters)

    @property
    def valid_when(self):
        return numba.types.boolean

    @property
    def parameters(self):
        return numba.types.StringLiteral(self._parameters)

    @property
    def mask(self):
        return ak.numba.GrowableBufferType(self._dtype)

    @property
    def content(self):
        return to_numbatype(self._content)

    @property
    def length(self):
        return numba.types.int64


@numba.extending.register_model(ByteMaskedType)
class ByteMaskedModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("mask", fe_type.mask),
            ("content", fe_type.content),
            ("valid_when", fe_type.valid_when),
        ]
        super().__init__(dmm, fe_type, members)


for member in (
    "mask",
    "content",
    "valid_when",
):
    numba.extending.make_attribute_wrapper(ByteMaskedType, member, "_" + member)


@numba.extending.unbox(ByteMaskedType)
def ByteMaskedType_unbox(typ, obj, c):
    # get PyObjects
    content_obj = c.pyapi.object_getattr_string(obj, "_content")
    mask_obj = c.pyapi.object_getattr_string(obj, "_mask")
    valid_when_obj = c.pyapi.object_getattr_string(obj, "_valid_when")

    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out.content = c.pyapi.to_native_value(typ.content, content_obj).value
    out.mask = c.pyapi.to_native_value(typ.mask, mask_obj).value
    out.valid_when = c.pyapi.to_native_value(typ.valid_when, valid_when_obj).value

    # decref PyObjects
    c.pyapi.decref(content_obj)
    c.pyapi.decref(mask_obj)
    c.pyapi.decref(valid_when_obj)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(ByteMaskedType)
def ByteMaskedType_box(typ, val, c):
    # get PyObject of the ByteMasked class
    ByteMasked_obj = c.pyapi.unserialize(c.pyapi.serialize_object(ByteMasked))

    builder = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    mask_obj = c.pyapi.from_native_value(typ.mask, builder.mask, c.env_manager)
    content_obj = c.pyapi.from_native_value(typ.content, builder.content, c.env_manager)

    out = c.pyapi.call_function_objargs(
        ByteMasked_obj,
        (
            mask_obj,
            content_obj,
        ),
    )

    # decref PyObjects
    c.pyapi.decref(ByteMasked_obj)

    c.pyapi.decref(mask_obj)
    c.pyapi.decref(content_obj)

    return out


@numba.extending.overload_method(ByteMaskedType, "_length_get", inline="always")
def ByteMasked_length(builder):
    def getter(builder):
        return len(builder._content)

    return getter


@numba.extending.overload_method(ByteMaskedType, "append")
def ByteMasked_append(builder, datum):
    if isinstance(builder, ByteMaskedType):

        def append(builder, datum):
            builder._mask.append(builder._valid_when)
            builder._content.append(datum)

        return append


@numba.extending.overload_method(ByteMaskedType, "extend")
def ByteMasked_extend(builder, data):
    def extend(builder, data):
        builder._mask.extend([builder._valid_when] * len(data))
        builder._content.extend(data)

    return extend


@numba.extending.overload_method(ByteMaskedType, "append_null")
def ByteMasked_append_null(builder):
    if isinstance(builder, ByteMaskedType):

        def append_null(builder):
            builder._mask.append(not builder._valid_when)
            builder._content.append(np.nan)

        return append_null


@numba.extending.overload_method(ByteMaskedType, "extend_null")
def ByteMasked_extend_null(builder, size):
    def extend_null(builder, size):
        builder._mask.extend([not builder._valid_when] * size)
        builder._content.extend([np.nan] * size)

    return extend_null


########## BitMasked #########################################################


@final
class BitMasked(LayoutBuilder):
    def __init__(
        self,
        dtype,  # mask must be "uint8"
        content,  # FIXME
        valid_when,
        lsb_order,
        *,
        parameters=None,
        initial=1024,
        resize=8.0,
    ):
        self._mask = GrowableBuffer(dtype=dtype, initial=initial, resize=resize)
        self._content = content
        self._valid_when = valid_when
        self._lsb_order = lsb_order
        self._current_byte_index = np.zeros((2,), dtype=np.uint8)
        # FIXME:
        # self._current_byte = np.uint8(0)
        # self._current_index = 0
        self._mask.append(self._current_byte_index[0])
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

    def __repr__(self):  # as constructor
        return f"ak.numba.lb.BitMasked({self._mask.dtype}, {self._content}, {self._valid_when}, {self._lsb_order}, parameters={self._parameters})"

    def type(self):
        return f"ak.numba.lb.BitMasked({self._mask.dtype}, {self._content}, {self._valid_when}, {self._lsb_order}, parameters={self._parameters})"

    def numbatype(self):
        return BitMaskedType(
            numba.from_dtype(self._mask.dtype),
            self.content,
            self.valid_when,
            self.lsb_order,
            self.parameters,
        )

    @property
    def content(self):
        return self._content

    @property
    def valid_when(self):
        return self._valid_when

    @property
    def lsb_order(self):
        return self._lsb_order

    def _append_begin(self):
        """
        Private helper function.
        """
        if self._current_byte_index[1] == 8:
            self._current_byte_index[0] = np.uint8(0)
            self._mask.append(self._current_byte_index[0])
            self._current_byte_index[1] = 0

    def _append_end(self):
        """
        Private helper function.
        """
        self._current_byte_index[1] += 1
        if self._valid_when:
            # 0 indicates null, 1 indicates valid
            self._mask._panels[-1][
                self._mask._length_pos[1] - 1
            ] = self._current_byte_index[0]
        else:
            # 0 indicates valid, 1 indicates null
            self._mask._panels[-1][
                self._mask._length_pos[1] - 1
            ] = ~self._current_byte_index[0]

    def append_valid(self):
        self._append_begin()
        # current_byte_ and cast_: 0 indicates null, 1 indicates valid
        self._current_byte_index[0] |= self._cast[self._current_byte_index[1]]
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

    @property
    def _length(self):
        return (
            len(self._mask)
            if len(self._mask) == 0
            else (len(self._mask) - 1) * 8 + self._current_byte_index[1]
        )

    def __len__(self):
        return self._length

    def is_valid(self, error: str):
        if len(self._content) != self._length:
            error = f"BitMasked has content length {len(self._content)} but bit mask length {self._length}"
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
                length=self._length,
                lsb_order=self._lsb_order,
                parameters=self._parameters,
            )
        )


class BitMaskedType(numba.types.Type):
    def __init__(self, dtype, content, valid_when, lsb_order, parameters):
        super().__init__(
            name=f"ak.numba.lb.BitMasked({dtype}, {content.type()}, {valid_when}, {lsb_order}, parameters={parameters.literal_value if isinstance(parameters, numba.types.Literal) else None})"
        )
        self._dtype = dtype
        self._content = content
        self._valid_when = valid_when
        self._lsb_order = lsb_order
        self._parameters = parameters

    @classmethod
    def type(cls):
        return BitMaskedType(
            cls.dtype, cls.content, cls.valid_when, cls.lsb_order, cls.parameters
        )

    @property
    def parameters(self):
        return numba.types.StringLiteral(self._parameters)

    @property
    def mask(self):
        return ak.numba.GrowableBufferType(self._dtype)

    @property
    def valid_when(self):
        return numba.types.boolean

    @property
    def lsb_order(self):
        return numba.types.boolean

    @property
    def current_byte_index(self):
        return numba.types.Array(numba.types.uint8, 1, "C")

    @property
    def content(self):
        return to_numbatype(self._content)

    @property
    def length(self):
        return numba.types.int64


@numba.extending.register_model(BitMaskedType)
class BitMaskedModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("mask", fe_type.mask),
            ("content", fe_type.content),
            ("valid_when", fe_type.valid_when),
            ("lsb_order", fe_type.lsb_order),
            ("current_byte_index", fe_type.current_byte_index),
        ]
        super().__init__(dmm, fe_type, members)


for member in (
    "mask",
    "content",
    "valid_when",
    "lsb_order",
    "current_byte_index",
):
    numba.extending.make_attribute_wrapper(BitMaskedType, member, "_" + member)


@numba.extending.unbox(BitMaskedType)
def BitMaskedType_unbox(typ, obj, c):
    # get PyObjects
    mask_obj = c.pyapi.object_getattr_string(obj, "_mask")
    content_obj = c.pyapi.object_getattr_string(obj, "_content")
    valid_when_obj = c.pyapi.object_getattr_string(obj, "_valid_when")
    lsb_order_obj = c.pyapi.object_getattr_string(obj, "_lsb_order")
    current_byte_index_obj = c.pyapi.object_getattr_string(obj, "_current_byte_index")

    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out.mask = c.pyapi.to_native_value(typ.mask, mask_obj).value
    out.content = c.pyapi.to_native_value(typ.content, content_obj).value
    out.valid_when = c.pyapi.to_native_value(typ.valid_when, valid_when_obj).value
    out.lsb_order = c.pyapi.to_native_value(typ.lsb_order, lsb_order_obj).value
    out.current_byte_index = c.pyapi.to_native_value(
        typ.current_byte_index, current_byte_index_obj
    ).value

    # decref PyObjects
    c.pyapi.decref(mask_obj)
    c.pyapi.decref(content_obj)
    c.pyapi.decref(valid_when_obj)
    c.pyapi.decref(lsb_order_obj)
    c.pyapi.decref(current_byte_index_obj)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(BitMaskedType)
def BitMaskedType_box(typ, val, c):
    # get PyObject of the BitMasked class
    BitMasked_obj = c.pyapi.unserialize(c.pyapi.serialize_object(BitMasked))

    builder = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    mask_obj = c.pyapi.from_native_value(typ.mask, builder.mask, c.env_manager)
    content_obj = c.pyapi.from_native_value(typ.content, builder.content, c.env_manager)
    valid_when_obj = c.pyapi.from_native_value(
        typ.valid_when, builder.valid_when, c.env_manager
    )
    lsb_order_obj = c.pyapi.from_native_value(
        typ.lsb_order, builder.lsb_order, c.env_manager
    )

    out = c.pyapi.call_function_objargs(
        BitMasked_obj,
        (
            mask_obj,
            content_obj,
            valid_when_obj,
            lsb_order_obj,
        ),
    )

    # decref PyObjects
    c.pyapi.decref(BitMasked_obj)

    c.pyapi.decref(mask_obj)
    c.pyapi.decref(content_obj)
    c.pyapi.decref(valid_when_obj)
    c.pyapi.decref(lsb_order_obj)

    return out


@numba.extending.overload_attribute(BitMaskedType, "_cast")
def BitMaskedType_cast(builder):
    def get_cast(builder):
        if builder._lsb_order:
            return np.array(
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
            return np.array(
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

    return get_cast


@numba.extending.overload_method(BitMaskedType, "_length_get", inline="always")
def BitMasked_length(builder):
    def getter(builder):
        return len(builder._content)

    return getter


@numba.extending.overload_method(BitMaskedType, "_append_begin", inline="always")
def BitMasked_append_begin(builder):
    def append_begin(builder):
        if builder._current_byte_index[1] == 8:
            builder._current_byte_index[0] = np.uint8(0)
            builder._mask.append(np.uint8(0))
            builder._current_byte_index[1] = 0

    return append_begin


@numba.extending.overload_method(BitMaskedType, "_append_end", inline="always")
def BitMasked_append_end(builder):
    def append_end(builder):
        builder._current_byte_index[1] += 1
        if builder._valid_when:
            # 0 indicates null, 1 indicates valid
            builder._mask._panels[-1][
                builder._mask._length_pos[1] - 1
            ] = builder._current_byte_index[0]
        else:
            # 0 indicates valid, 1 indicates null
            builder._mask._panels[-1][
                builder._mask._length_pos[1] - 1
            ] = ~builder._current_byte_index[0]

    return append_end


@numba.extending.overload_method(BitMaskedType, "append")
def BitMasked_append(builder, datum):
    if isinstance(builder, BitMaskedType):

        def append(builder, datum):
            builder._append_begin()
            # current_byte_ and cast_: 0 indicates null, 1 indicates valid
            builder._current_byte_index[0] |= builder._cast[
                builder._current_byte_index[1]
            ]
            builder._append_end()
            builder._content.append(datum)

        return append


@numba.extending.overload_method(BitMaskedType, "extend")
def BitMasked_extend(builder, data):
    def extend(builder, data):
        for _ in range(len(data)):
            builder.append_valid()
        builder._content.extend(data)

    return extend


@numba.extending.overload_method(BitMaskedType, "append_valid")
def BitMasked_append_valid(builder):
    def append_valid(builder):
        builder._append_begin()
        # current_byte_ and cast_: 0 indicates null, 1 indicates valid
        builder._current_byte_index[0] |= builder._cast[builder._current_byte_index[1]]
        builder._append_end()

    return append_valid


@numba.extending.overload_method(BitMaskedType, "append_null")
def BitMasked_append_null(builder):
    if isinstance(builder, BitMaskedType):

        def append_null(builder):
            builder._append_begin()
            # current_byte_ and cast_ default to null, no change
            builder._append_end()
            builder._content.append(np.nan)

        return append_null


@numba.extending.overload_method(BitMaskedType, "extend_null")
def BitMasked_extend_null(builder, size):
    def extend_null(builder, size):
        for _ in range(size):
            builder.append_null()
        # builder._content.extend([np.nan] * size)

    return extend_null


########## Unmasked #########################################################


@final
class Unmasked(LayoutBuilder):
    def __init__(self, content, *, parameters=None):
        self._content = content
        self._parameters = parameters

    def __repr__(self):
        return f"ak.numba.lb.Unmasked({self._content}, parameters={self._parameters})"

    def type(self):
        return f"ak.numba.lb.Unmasked({self._content}, parameters={self._parameters})"

    def numbatype(self):
        return UnmaskedType(
            self.content,
            numba.types.StringLiteral(self._parameters),
        )

    @property
    def content(self):
        return self._content

    # FIXME: what if content does not have append?
    def append(self, value):
        self._content.append(value)

    # FIXME: what if content does not have extend?
    def extend(self, data):
        return self._content.extend(data)

    @property
    def parameters(self):
        return self._parameters

    def clear(self):
        self._content.clear()

    @property
    def _length(self):
        return len(self._content)

    def __len__(self):
        return self._length

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


class UnmaskedType(numba.types.Type):
    def __init__(self, content, parameters):
        super().__init__(
            name=f"ak.numba.lb.Unmasked({content.type()}, parameters={parameters.literal_value if isinstance(parameters, numba.types.Literal) else None})"
        )
        self._content = content

    @classmethod
    def type(cls):
        return UnmaskedType(cls.content, cls.parameters)

    @property
    def parameters(self):
        return numba.types.StringLiteral(self._parameters)

    @property
    def content(self):
        return to_numbatype(self._content)

    @property
    def length(self):
        return numba.types.int64


@numba.extending.register_model(UnmaskedType)
class UnmaskedModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("content", fe_type.content),
        ]
        super().__init__(dmm, fe_type, members)


for member in ("content",):
    numba.extending.make_attribute_wrapper(UnmaskedType, member, "_" + member)


@numba.extending.unbox(UnmaskedType)
def UnmaskedType_unbox(typ, obj, c):
    # get PyObjects
    content_obj = c.pyapi.object_getattr_string(obj, "_content")

    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out.content = c.pyapi.to_native_value(typ.content, content_obj).value

    # decref PyObjects
    c.pyapi.decref(content_obj)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(UnmaskedType)
def UnmaskedType_box(typ, val, c):
    # get PyObject of the Unmasked class
    Unmasked_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Unmasked))

    builder = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    content_obj = c.pyapi.from_native_value(typ.content, builder.content, c.env_manager)

    out = c.pyapi.call_function_objargs(
        Unmasked_obj,
        (content_obj,),
    )

    # decref PyObjects
    c.pyapi.decref(Unmasked_obj)

    c.pyapi.decref(content_obj)

    return out


@numba.extending.overload_method(UnmaskedType, "_length_get", inline="always")
def Unmasked_length(builder):
    def getter(builder):
        return len(builder._content)

    return getter


@numba.extending.overload_method(UnmaskedType, "append")
def Unmasked_append(builder, datum):
    if isinstance(builder, UnmaskedType):

        def append(builder, datum):
            builder._content.append(datum)

        return append


@numba.extending.overload_method(UnmaskedType, "extend")
def Unmasked_extend(builder, data):
    if isinstance(builder, UnmaskedType):

        def extend(builder, data):
            builder._content.extend(data)

        return extend


########## Record #########################################################


@final
class Record(LayoutBuilder):
    def __init__(self, contents, fields, *, parameters=None):
        assert len(fields) != 0
        self._contents = tuple(contents)
        self._fields = tuple(fields)
        self._parameters = parameters

    @property
    def contents(self):
        return self._contents

    @property
    def fields(self):
        return self._fields

    def __repr__(self):
        return f"ak.numba.lb.Record({self.contents}, {self.fields}, parameters={self._parameters})"

    def type(self):
        return f"ak.numba.lb.Record({self.contents}, {self.fields}, parameters={self._parameters})"

    def numbatype(self):
        return RecordType(
            self.contents,
            self.fields,
            numba.types.StringLiteral(self._parameters),
        )

    def content(self, name):
        return self._contents[self._fields.index(name)]

    def field(self, name):
        return self.content(name)

    def parameters(self):
        return self._parameters

    def clear(self):
        for content in self._contents:
            content.clear()

    @property
    def _length(self):
        return len(self._contents[0])

    def __len__(self):
        return self._length

    def is_valid(self, error: str):
        length = -1
        for i, content in enumerate(self._contents):
            if length == -1:
                length = len(content)
            elif length != len(content):
                error = f"Record has field {self._fields[i]} length {len(content)} that differs from the first length {length}"
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


class RecordType(numba.types.Type):
    def __init__(self, contents, fields, parameters):
        super().__init__(
            name=f"ak.numba.lb.Record({contents}, {fields}, parameters={parameters.literal_value if isinstance(parameters, numba.types.Literal) else None})"
        )
        self._contents = contents
        self._fields = fields
        self._parameters = parameters

    @classmethod
    def type(cls):
        return RecordType(cls.contents, cls.fields, cls.parameters)

    @property
    def parameters(self):
        return numba.types.StringLiteral(self._parameters)

    @property
    def contents(self):
        return numba.types.Tuple([to_numbatype(it) for it in self._contents])

    @property
    def fields(self):
        return numba.types.Tuple(
            to_numbatype([numba.types.StringLiteral(it) for it in self._fields])
        )

    # def content(self, name):  # Literal string or Literal int
    #     return to_numbatype(self._contents[self._fields.index(name)])
    #
    # def field(self, name):  # Literal string or Literal int
    #     return to_numbatype(self._contents[self._fields.index(name)])
    #
    @property
    def length(self):
        return numba.types.int64


@numba.extending.register_model(RecordType)
class RecordModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("contents", fe_type.contents),
            ("fields", fe_type.fields),
        ]
        super().__init__(dmm, fe_type, members)


for member in (
    "contents",
    "fields",
):
    numba.extending.make_attribute_wrapper(RecordType, member, "_" + member)


@numba.extending.unbox(RecordType)
def RecordType_unbox(typ, obj, c):
    # get PyObjects
    contents_obj = c.pyapi.object_getattr_string(obj, "_contents")
    fields_obj = c.pyapi.object_getattr_string(obj, "_fields")

    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out.contents = c.pyapi.to_native_value(typ.contents, contents_obj).value
    out.fields = c.pyapi.to_native_value(typ.fields, fields_obj).value

    # decref PyObjects
    c.pyapi.decref(contents_obj)
    c.pyapi.decref(fields_obj)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(RecordType)
def RecordType_box(typ, val, c):
    # get PyObject of the Record class
    Record_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Record))

    builder = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    contents_obj = c.pyapi.from_native_value(
        typ.contents, builder.contents, c.env_manager
    )
    fields_obj = c.pyapi.from_native_value(typ.fields, builder.fields, c.env_manager)

    out = c.pyapi.call_function_objargs(
        Record_obj,
        (
            contents_obj,
            fields_obj,
        ),
    )

    # decref PyObjects
    c.pyapi.decref(Record_obj)

    c.pyapi.decref(contents_obj)
    c.pyapi.decref(fields_obj)

    return out


@numba.extending.overload_method(RecordType, "_length_get", inline="always")
def Record_length(builder):
    def getter(builder):
        return len(builder._contents[0])

    return getter


@numba.extending.overload_method(RecordType, "_field_index")
def Record_field_index(builder, name):
    if isinstance(builder, RecordType):

        def getter(builder, name):
            return builder._fields.index(name)

        return getter


@numba.extending.overload_method(RecordType, "content")
def Record_content(builder, name):
    if isinstance(builder, RecordType):

        def getter(builder, name):
            # FIXME: Tuple needs a compile-time index
            content = builder._contents[builder._fields.index(name)]

            return content

        return getter


########## Tuple #######################################################


@final
class Tuple(LayoutBuilder):
    def __init__(self, contents, *, parameters=None):
        assert len(contents) != 0
        self._contents = contents
        self._parameters = parameters

    def index(self, at):
        return self._contents[at]

    def parameters(self):
        return self._parameters

    def clear(self):
        for _content in self._contents:
            _content.clear()

    def length(self):
        return len(self._contents[0])

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


class TupleType(numba.types.Type):
    def __init__(self, contents):
        super().__init__(name=f"ak.numba.lb.Tuple({contents})")
        self._contents = contents

    @classmethod
    def type(cls):
        return TupleType(cls.contents)

    def __repr__(self):
        return f"<Tuple of {self._contents!r}>"

    @property
    def parameters(self):
        return numba.types.StringLiteral

    @property
    def index(self, at):
        return to_numbatype(self._contents[at])

    @property
    def length(self):
        return numba.types.int64


@numba.extending.register_model(TupleType)
class TupleModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            # ("contents", fe_type.contents),
        ]
        super().__init__(dmm, fe_type, members)


# for member in ("contents",):
#     numba.extending.make_attribute_wrapper(TupleType, member, "_" + member)


@numba.extending.unbox(TupleType)
def TupleType_unbox(typ, obj, c):
    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


########## Union #######################################################


@final
class Union(LayoutBuilder):
    def __init__(
        self,
        dtype,
        contents,
        *,
        parameters=None,
        initial=1024,
        resize=8.0,
    ):
        self._last_valid_index = [-1] * len(contents)
        self._tags = GrowableBuffer("int8", initial=initial, resize=resize)
        self._index = GrowableBuffer(dtype=dtype, initial=initial, resize=resize)
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


class UnionType(numba.types.Type):
    def __init__(self, dtype, contents):
        super().__init__(name="ak.numba.lb.Union()")
        self._dtype = dtype
        self._contents = contents

    @classmethod
    def type(cls):
        return UnionType(cls._is_tuple)

    @property
    def parameters(self):
        return numba.types.StringLiteral

    @property
    def length(self):
        return numba.types.int64


@numba.extending.register_model(UnionType)
class UnionModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            # ("contents", fe_type.contents),
        ]
        super().__init__(dmm, fe_type, members)


# for member in ("contents", ):
#     numba.extending.make_attribute_wrapper(UnionType, member, "_" + member)


@numba.extending.unbox(UnionType)
def UnionType_unbox(typ, obj, c):
    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)