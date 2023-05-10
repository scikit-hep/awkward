# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import json  # FIXME:

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

    def form(self):
        return self._content.form()

    def is_valid(self, error: str):
        return self._content.is_valid(error)

    # def snapshot(self, *, highlevel=True, behavior=None) -> ArrayLike:
    #     return self._content.snapshot()

    def _type(self, typestrs):
        raise NotImplementedError


########## Numpy ############################################################


@final
class Numpy(LayoutBuilder):
    def __init__(self, dtype, *, parameters=None, initial=1024, resize=8.0):
        self._dtype = np.dtype(dtype)
        self._data = GrowableBuffer(dtype=dtype, initial=initial, resize=resize)
        self._parameters = parameters
        self._id = 0

    @classmethod
    def _from_buffer(cls, data):
        out = cls.__new__(cls)
        out._data = data  # GrowableBuffer(dtype=dtype, initial=initial, resize=resize)
        out._parameters = ""  # FIXME: parameters?
        out._id = 0
        return out

    @property
    def dtype(self):
        return self._data.dtype

    def __repr__(self):
        return f"<Numpy of {self.dtype!r} with {self._length} items>"

    def _type(self, typestrs):
        ...

    @property
    def _length(self):
        return len(self._data)

    def __len__(self):
        return self._length

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id: int):
        self._id = id
        id += 1
        return id

    # FIXME: LayoutBuilder.id = (next_id := LayoutBuilder.id)

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
        return ak.from_buffers(
            self.form(), self._length, {f"node{self.id}-data": self._data.snapshot()}
        )

    def form(self):
        # FIXME: no numba
        form_key = f"node{self.id}"
        params = ""
        if self._parameters is not None:
            params = (
                "" if self._parameters == "" else f", parameters: {self._parameters}"
            )

        return f'{{"class": "NumpyArray", "primitive": "{self._data.dtype}", "form_key": "{form_key}"{params}}}'


class NumpyType(numba.types.Type):
    def __init__(self, dtype):
        super().__init__(name=f"ak.numba.lb.Numpy({dtype})")
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def parameters(self):
        return numba.types.StringLiteral

    @property
    def data(self):
        return ak.numba.GrowableBufferType(self._dtype)

    @property
    def length(self):
        return numba.types.float64


@numba.extending.typeof_impl.register(Numpy)
def typeof_Numpy(val, c):
    return NumpyType(numba.from_dtype(val._data.dtype))


@numba.extending.register_model(NumpyType)
class NumpyModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("dtype", fe_type.data.dtype),
            ("data", fe_type.data),
        ]
        super().__init__(dmm, fe_type, members)


for member in (
    "dtype",
    "data",
):
    numba.extending.make_attribute_wrapper(NumpyType, member, "_" + member)


# @numba.extending.overload_attribute(NumpyType, "dtype")
# def NumpyType_dtype(builder):
#     def getter(builder):
#         if isinstance(builder, numba.types.StringLiteral):
#         return builder._data.dtype
#
#     return getter


@numba.extending.unbox(NumpyType)
def NumpyType_unbox(typ, obj, c):
    # get PyObjects
    # dtype_obj = c.pyapi.object_getattr_string(obj, "_dtype")
    data_obj = c.pyapi.object_getattr_string(obj, "_data")

    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    # out.dtype = c.pyapi.to_native_value(typ.dtype, data_obj).value
    out.data = c.pyapi.to_native_value(typ.data, data_obj).value

    # decref PyObjects
    # c.pyapi.decref(dtype_obj)
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
    # dtype_obj = c.pyapi.from_native_value(typ.dtype, builder.dtype, c.env_manager)
    data_obj = c.pyapi.from_native_value(typ.data, builder.data, c.env_manager)

    out = c.pyapi.call_function_objargs(
        from_buffer_obj,
        (data_obj,),
    )

    # decref PyObjects
    c.pyapi.decref(Numpy_obj)
    c.pyapi.decref(from_buffer_obj)

    # c.pyapi.decref(dtype_obj)
    c.pyapi.decref(data_obj)

    return out


def _from_buffer():
    ...


@numba.extending.type_callable(_from_buffer)
def Numpy_from_buffer_typer(context):
    def typer(buffer):
        if isinstance(buffer, GrowableBufferType):
            return NumpyType(buffer)

    return typer


@numba.extending.lower_builtin(_from_buffer, GrowableBufferType)
def Numpy_from_buffer_impl(context, builder, sig, args):
    out = numba.core.cgutils.create_struct_proxy(sig.return_type)(context, builder)
    out.data = args[0]

    if context.enable_nrt:
        context.nrt.incref(builder, sig.args[0], args[0])

    return out._getvalue()


@numba.extending.overload(Numpy)
def Numpy_ctor(dtype):  # , parameters=None, initial=1024, resize=8.0):
    if isinstance(dtype, numba.types.StringLiteral):
        dt = np.dtype(dtype.literal_value)

    elif isinstance(dtype, numba.types.DTypeSpec):
        dt = numba.core.typing.npydecl.parse_dtype(dtype)

    else:
        return

    def ctor_impl(dtype):  # , parameters=None, initial=1024, resize=8.0):
        # panels = numba.typed.List([np.empty((initial,), dt)])
        # length_pos = np.zeros((2,), dtype=np.int64)
        # data = ak.numba._from_data(panels, length_pos, resize)
        return NumpyType(dt)

    return ctor_impl


@numba.extending.overload_method(NumpyType, "_length_get", inline="always")
def Numpy_length(builder):
    def getter(builder):
        return builder._data._length_pos[0]

    return getter


@numba.extending.overload(len)
def NumpyType_len(builder):
    if isinstance(builder, NumpyType):

        def len_impl(builder):
            return builder._length_get()

        return len_impl


@numba.extending.overload_method(NumpyType, "append")
def Numpy_append(builder, datum):
    def append(builder, datum):
        buffer = builder._data
        buffer.append(datum)

    return append


@numba.extending.overload_method(NumpyType, "extend")
def Numpy_extend(builder, data):
    def extend(builder, data):
        builder._data.extend(data)

    return extend


@numba.extending.overload_method(NumpyType, "snapshot")
def Numpy_snapshot(builder):
    def snapshot(builder):
        return builder._data.snapshot()

    return snapshot


########## Empty ############################################################


@final
class Empty(LayoutBuilder):
    def __init__(self, *, parameters=None):
        self._parameters = parameters
        self._id = 0

    @classmethod
    def _from_buffer(cls):
        out = cls.__new__(cls)
        out._parameters = ""  # FIXME: parameters?
        return out

    def __repr__(self):
        return f"<Empty with {self.length} items>"

    def _type(self, typestrs):
        ...

    @property
    def _length(self):
        return 0

    def length(self):
        return self._length

    def __len__(self):
        return self._length

    def parameters(self):
        return self._parameters

    def set_id(self, id: int):
        self._id = id
        id += 1

    def clear(self):
        pass

    def is_valid(self, error: str):
        return True

    def form(self):
        params = ""
        if self._parameters is not None:
            params = (
                "" if self._parameters == "" else f', "parameters": {self._parameters}'
            )
        return f'{{"class": "EmptyArray"{params}}}'

    def snapshot(self) -> ArrayLike:
        return ak.from_buffers(self.form(), len(self), {})


class EmptyType(numba.types.Type):
    def __init__(self):
        super().__init__(name="ak.numba.lb.Empty()")

    @property
    def length(self):
        return numba.types.int64


@numba.extending.typeof_impl.register(Empty)
def typeof_Empty(val, c):
    return EmptyType()


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
    from_data_obj = c.pyapi.object_getattr_string(Empty_obj, "_from_buffer")

    out = c.pyapi.call_function_objargs(from_data_obj, ())

    # decref PyObjects
    c.pyapi.decref(Empty_obj)
    c.pyapi.decref(from_data_obj)

    return out


@numba.extending.overload(Empty)
def Empty_ctor():
    def ctor_impl():
        return _from_buffer()

    return ctor_impl


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


@numba.extending.overload(len)
def Empty_len(builder):
    if isinstance(builder, EmptyType):

        def len_impl(builder):
            return builder._length_get()

        return len_impl


########## ListOffset #########################################################


@final
class ListOffset(LayoutBuilder):
    def __init__(self, dtype, content, *, parameters=None, initial=1024, resize=8.0):
        self._offsets = GrowableBuffer(dtype=dtype, initial=initial, resize=resize)
        self._offsets.append(0)
        self._content = content
        self._parameters = parameters
        self._id = 0

    def __repr__(self):
        return f"<ListOffset of {self._content!r} with {self._length} items>"

    @property
    def content(self):
        return self._content

    def begin_list(self):
        return self._content

    def end_list(self):
        self._offsets.append(self._content._length)

    def parameters(self):
        return self._parameters

    def set_id(self, id: int):
        self._id = id
        id += 1
        self._content.set_id(id)

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

    def buffer_nbytes(self, names_nbytes):
        names_nbytes[f"node{self._id}-offsets"] = self._offsets.nbytes()
        self._content.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self._offsets.concatenate(buffers[f"node{self._id}-offsets"])
        self._content.to_buffers(buffers)

    def form(self):
        params = "" if self._parameters == "" else f", parameters: {self._parameters}"
        return f'{{"class": "ListOffsetArray", "offsets": "{self._offsets.dtype}", "content": {self._content.form()}, "form_key": "node{self._id}"{params}}}'

    def snapshot(self) -> ArrayLike:
        """
        Converts the currently accumulated data into an #ak.Array.
        """
        content = self._content.snapshot()

        return ak.Array(
            ak.contents.listoffsetarray.ListOffsetArray(
                ak.index.Index(self._offsets.snapshot()),
                content.layout,
            )
        )


########## List ############################################################


@final
class List(LayoutBuilder):
    def __init__(self, dtype, content, *, parameters=None, initial=1024, resize=8.0):
        self._starts = GrowableBuffer(dtype=dtype, initial=initial, resize=resize)
        self._stops = GrowableBuffer(dtype=dtype, initial=initial, resize=resize)
        self._content = content
        self._parameters = parameters
        self._id = 0

    @property
    def content(self):
        return self._content

    def begin_list(self):
        self._starts.append(self._content._length)
        return self._content

    def end_list(self):
        self._stops.append(self._content._length)

    def parameters(self):
        return self._parameters

    def set_id(self, id: int):
        self._id = id
        id += 1
        self._content.set_id(id)

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

    def buffer_nbytes(self, names_nbytes):
        names_nbytes[f"node{self._id}-starts"] = self._starts.nbytes()
        names_nbytes[f"node{self._id}-stops"] = self._stops.nbytes()
        self._content.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self._starts.concatenate(buffers[f"node{self._id}-starts"])
        self._stops.concatenate(buffers[f"node{self._id}-stops"])
        self._content.to_buffers(buffers)

    def form(self):
        params = "" if self._parameters == "" else f", parameters: {self._parameters}"
        return f'{{"class": "ListArray", "starts": "{self._starts.index_form()}", "stops": "{self._stops.index_form()}", "content": {self._content.form()}, "form_key": "node{self._id}"{params}}}'

    def snapshot(self) -> ArrayLike:
        """
        Converts the currently accumulated data into an #ak.Array.
        """
        return ak.Array(
            ak.contents.ListArray(
                ak.index.Index(self._starts.snapshot()),
                ak.index.Index(self._stops.snapshot()),
                self._content.snapshot().layout,
            )
        )


########## Regular ############################################################


@final
class Regular(LayoutBuilder):
    def __init__(self, content, size, *, parameters=None):
        self._length = 0
        self._content = content
        self._size = size
        self._parameters = parameters
        self._id = 0

    @property
    def content(self):
        return self._content

    def begin_list(self):
        return self._content

    def end_list(self):
        self._length += 1

    def parameters(self):
        return self._parameters

    def set_id(self, id: int):
        self._id = id
        id += 1
        self._content.set_id(id)

    def clear(self):
        self._content.clear()

    def length(self):
        return self._length

    def __len__(self):
        return self._length

    def is_valid(self, error: str):
        if self._content._length != self._length * self._size:
            error = f"Regular node{self._id} has content length {self._content._length}, but length {self._length} and size {self._size}"
            return False
        else:
            return self._content.is_valid(error)

    def buffer_nbytes(self, names_nbytes):
        self._content.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self._content.to_buffers(buffers)

    def form(self):
        params = "" if self._parameters == "" else f", parameters: {self._parameters}"
        return f'{{"class": "RegularArray", "size": {self._size}, "content": {self._content.form()}, "form_key": "node{self._id}"{params}}}'

    def snapshot(self) -> ArrayLike:
        """
        Converts the currently accumulated data into an #ak.Array.
        """
        return ak.Array(
            ak.contents.RegularArray(
                self._content.snapshot().layout,
                self._size,
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
        self._id = 0

    @property
    def content(self):
        return self._content

    def append_index(self):
        self._last_valid = self._content._length
        self._index.append(self._last_valid)
        return self._content

    def extend_index(self, size):
        start = self._content._length
        stop = start + size
        self._last_valid = stop - 1
        self._index.extend(list(range(start, stop)))
        return self._content

    def parameters(self):
        return self._parameters

    def set_id(self, id: int):
        self._id = id
        id += 1
        self._content.set_id(id)

    def clear(self):
        self._last_valid = -1
        self._index.clear()
        self._content.clear()

    def length(self):
        return self._index._length

    def __len__(self):
        return self.length()

    def is_valid(self, error: str):
        if self._content.length() != self._index.length():
            error = f"Indexed node{self._id} has content length {self._content.length()} but index length {self._index.length()}"
            return False
        elif self._content.length() != self._last_valid + 1:
            error = f"Indexed node{self._id} has content length {self._content.length()} but last valid index is {self._last_valid}"
        else:
            return self._content.is_valid(error)

    def buffer_nbytes(self, names_nbytes):
        names_nbytes[f"node{self._id}-index"] = self._index.nbytes()
        self._content.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self._index.concatenate(buffers[f"node{self._id}-index"])
        self._content.to_buffers(buffers)

    def form(self):
        params = "" if self._parameters == "" else f", parameters: {self._parameters}"
        return f'{{"class": "IndexedArray", "index": "{self._index.index_form()}", "content": {self._content.form()}, "form_key": "node{self._id}"{params}}}'

    def snapshot(self) -> ArrayLike:
        """
        Converts the currently accumulated data into an #ak.Array.
        """
        return ak.Array(
            ak.contents.IndexedArray(
                ak.index.Index64(self._index.snapshot()),
                self._content.snapshot().layout,
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
        self._id = 0

    @property
    def content(self):
        return self._content

    def append_index(self):
        self._last_valid = len(self._content)
        self._index.append(self._last_valid)
        return self._content

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

    def set_id(self, id: int):
        self._id = id
        id += 1
        self._content.set_id(id)

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
            error = f"Indexed node{self._id} has content length {self._content.length()} but last valid index is {self._last_valid}"
            return False
        else:
            return self._content.is_valid(error)

    def buffer_nbytes(self, names_nbytes):
        names_nbytes[f"node{self._id}-index"] = self._index.nbytes()
        self._content.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self._index.concatenate(buffers[f"node{self._id}-index"])
        self._content.to_buffers(buffers)

    def form(self):
        params = "" if self._parameters == "" else f", parameters: {self._parameters}"
        return f'{{"class": "IndexedOptionArray", "index": "{self._index.index_form()}", "content": {self._content.form()}, "form_key": "node{self._id}"{params}}}'

    def snapshot(self) -> ArrayLike:
        """
        Converts the currently accumulated data into an #ak.Array.
        """
        return ak.Array(
            ak.contents.IndexedOptionArray(
                ak.index.Index64(self._index.snapshot()),
                self._content.snapshot().layout,
            )
        )


########## ByteMasked #########################################################


@final
class ByteMasked(LayoutBuilder):
    def __init__(self, content, valid_when, parameters):
        self._mask = GrowableBuffer("int8")
        self._content = content
        self._valid_when = valid_when
        self._parameters = parameters
        self._id = 0

    @property
    def content(self):
        return self._content

    def valid_when(self):
        return self._valid_when

    def append_valid(self):
        self._mask.append(self._valid_when)
        return self._content

    def extend_valid(self, size):
        self._mask.extend([self._valid_when] * size, size)
        return self._content

    def append_null(self):
        self._mask.append(not self._valid_when)
        return self._content

    def extend_null(self, size):
        self._mask.extend([not self._valid_when] * size, size)
        return self._content

    def parameters(self):
        return self._parameters

    def set_id(self, id: int):
        self._id = id
        id += 1
        self._content.set_id(id)

    def clear(self):
        self._mask.clear()
        self._content.clear()

    def length(self):
        return self._mask.length()

    def is_valid(self, error: str):
        if self._content.length() != self._mask.length():
            error = f"ByteMasked node{self._id} has content length {self._content.length()} but mask length {len(self._stops)}"
            return False
        else:
            return self._content.is_valid(error)

    def buffer_nbytes(self, names_nbytes):
        names_nbytes[f"node{self._id}-mask"] = self._mask.nbytes()
        self._content.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self._mask.concatenate(buffers[f"node{self._id}-mask"])
        self._content.to_buffers(buffers)

    def form(self):
        params = "" if self._parameters == "" else f", parameters: {self._parameters}"
        return f'{{"class": "ByteMaskedArray", "mask": "{self._mask.index_form()}", "valid_when": {json.dumps(self._valid_when)}, "content": {self._content.form()}, "form_key": "node{self._id}"{params}}}'


########## BitMasked #########################################################


@final
class BitMasked(LayoutBuilder):
    def __init__(self, content, valid_when, lsb_order, parameters):
        self._mask = GrowableBuffer("uint8")
        self._content = content
        self._valid_when = valid_when
        self._lsb_order = lsb_order
        self._current_byte = np.uint8(0)
        self._current_byteref = self._mask.append_and_get_ref(self._current_byte)
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
        self._id = 0

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
            self._current_byteref = self._mask.append_and_get_ref(self._current_byte)
            self._current_index = 0

    def _append_end(self):
        """
        Private helper function.
        """
        self._current_index += 1
        if self._valid_when:
            # 0 indicates null, 1 indicates valid
            self._current_byteref.value = self._current_byte
        else:
            # 0 indicates valid, 1 indicates null
            self._current_byteref.value = ~self._current_byte

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

    def set_id(self, id: int):
        self._id = id
        id += 1
        self._content.set_id(id)

    def clear(self):
        self._mask.clear()
        self._content.clear()

    def length(self):
        return (self._mask.length() - 1) * 8 + self._current_index

    def is_valid(self, error: str):
        if self._content.length() != self.length():
            error = f"BitMasked node{self._id} has content length {self._content.length()} but bit mask length {self.length()}"
            return False
        else:
            return self._content.is_valid(error)

    def buffer_nbytes(self, names_nbytes):
        names_nbytes[f"node{self._id}-mask"] = self._mask.nbytes()
        self._content.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self._mask.concatenate(buffers[f"node{self._id}-mask"])
        self._content.to_buffers(buffers)

    def form(self):
        params = "" if self._parameters == "" else f", parameters: {self._parameters}"
        return f'{{"class": "BitMaskedArray", "mask": "{self._mask.index_form()}", "valid_when": {json.dumps(self._valid_when)}, "lsb_order": {json.dumps(self._lsb_order)}, "content": {self._content.form()}, "form_key": "node{self._id}"{params}}}'


########## Unmasked #########################################################


@final
class Unmasked(LayoutBuilder):
    def __init__(self, content, parameters):
        self._content = content
        self._parameters = parameters
        self._id = 0

    @property
    def content(self):
        return self._content

    def append_valid(self):
        return self._content

    def extend_valid(self, size):
        return self._content

    def parameters(self):
        return self._parameters

    def set_id(self, id: int):
        self._id = id
        id += 1
        self._content.set_id(id)

    def clear(self):
        self._content.clear()

    def length(self):
        return self._content.length()

    def is_valid(self, error: str):
        return self._content.is_valid(error)

    def buffer_nbytes(self, names_nbytes):
        self._content.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self._content.to_buffers(buffers)

    def form(self):
        params = "" if self._parameters == "" else f", parameters: {self._parameters}"
        return f'{{"class": "UnmaskedArray", "content": {self._content.form()}, "form_key": "node{self._id}"{params}}}'


########## Record #########################################################


class FieldPair:
    def __init__(self, name, content):
        self.name = name
        self.content = content


@final
class Record(LayoutBuilder):
    def __init__(self, field_pairs, *, parameters=None):
        assert len(field_pairs) != 0
        self._field_pairs = field_pairs
        self._first_content = field_pairs[0].content
        self._parameters = parameters
        self._id = 0

    def field(self, name):
        return self._field_pairs[name].content

    def parameters(self):
        return self._parameters

    def set_id(self, id: int):
        self._id = id
        id += 1
        for pair in self._field_pairs.values():
            pair.content.set_id(id)

    def clear(self):
        for pair in self._field_pairs.values():
            pair.content.clear()

    def length(self):
        return self._first_content.length()

    def is_valid(self, error: str):
        length = -1
        for pair in self._field_pairs.values():
            if length == -1:
                length = pair.content.length()
            elif length != pair.content.length():
                error = f"Record node{self._id} has field {pair.name} length {pair.content.length()} that differs from the first length {length}"
                return False
        for pair in self._field_pairs.values():
            if not pair.content.is_valid(error):
                return False
        return True

    def buffer_nbytes(self, names_nbytes):
        for pair in self._field_pairs.values():
            pair.content.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        for pair in self._field_pairs.values():
            pair.content.to_buffers(buffers)

    def form(self):
        params = "" if self._parameters == "" else f", parameters: {self._parameters}"
        pairs = ", ".join(
            f"{json.dumps(pair.name)}: {pair.content.form()}"
            for pair in self._field_pairs.values()
        )
        return f'{{"class": "RecordArray", "contents": {{{pairs}}}, "form_key": "node{self._id}"{params}}}'


########## Tuple #######################################################


@final
class Tuple(LayoutBuilder):
    def __init__(self, contents, parameters):
        assert len(contents) != 0
        self._contents = contents
        self._first_content = contents[0]
        self._parameters = parameters
        self._id = 0

    def index(self, at):
        return self._contents[at]

    def parameters(self):
        return self._parameters

    def set_id(self, id: int):
        self._id = id
        id += 1
        for content in self._contents:
            content.set_id(id)

    def clear(self):
        for _content in self._contents:
            _content.clear()

    def length(self):
        return self._first_content.length()

    def is_valid(self, error: str):
        length = -1
        for index, content in enumerate(self._contents):
            if length == -1:
                length = content.length()
            elif length != content.length():
                error = f"Tuple node{self._id} has index {index} length {content.length()} that differs from the first length {length}"
                return False
        for content in self._contents:
            if not content.is_valid(error):
                return False
        return True

    def buffer_nbytes(self, names_nbytes):
        for content in self._contents:
            content.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        for content in self._contents:
            content.to_buffers(buffers)

    def form(self):
        params = "" if self._parameters == "" else f", parameters: {self._parameters}"
        contents = ", ".join(content.form() for content in self._contents)
        return f'{{"class": "RecordArray", "contents": [{contents}], "form_key": "node{self._id}"{params}}}'


########## EmptyRecord #######################################################


@final
class EmptyRecord(LayoutBuilder):
    def __init__(self, is_tuple, parameters):
        self._length = 0
        self._is_tuple = is_tuple
        self._parameters = parameters
        self._id = 0

    def append(self):
        self._length += 1

    def extend(self, size):
        self._length += size

    def parameters(self):
        return self._parameters

    def set_id(self, id: int):
        self._id = id

    def clear(self):
        self._length = 0

    def length(self):
        return self._length

    def is_valid(self, error: str):
        return True

    def buffer_nbytes(self, names_nbytes):
        pass

    def to_buffers(self, buffers):
        pass

    def form(self):
        params = "" if self._parameters == "" else f", parameters: {self._parameters}"
        if self._is_tuple:
            return f'{{"class": "RecordArray", "contents": [], "form_key": "node{self._id}"{params}}}'
        else:
            return f'{{"class": "RecordArray", "contents": {{}}, "form_key": "node{self._id}"{params}}}'


########## Union #######################################################


@final
class Union(LayoutBuilder):
    def __init__(self, PRIMITIVE, contents, parameters):
        self._last_valid_index = [-1] * len(contents)
        self._tags = GrowableBuffer("int8")
        self._index = GrowableBuffer(PRIMITIVE)
        self._contents = contents
        self._parameters = parameters
        self._id = 0

    def append_index(self, tag):
        which_content = self._contents[tag]
        next_index = which_content.length()
        self._last_valid_index[tag] = next_index
        self._tags.append(tag)
        self._index.append(next_index)
        return which_content

    def parameters(self):
        return self._parameters

    def set_id(self, id: int):
        self._id = id
        id += 1
        for content in self._contents:
            content.set_id(id)

    def clear(self):
        for tag, _value in self._last_valid_index:
            self._last_valid_index[tag] = -1
        self._tags.clear()
        self._index.clear()
        for content in self._contents:
            content.clear()

    def length(self):
        return self._tags.length()

    def is_valid(self, error: str):
        for tag, _value in self._last_valid_index:
            if self._contents[tag].length() != self._last_valid_index[tag] + 1:
                error = f"Union node{self._id} has content {tag} length {self._contents[tag].length()} but last valid index is {self._last_valid_index[tag]}"
                return False
        for content in self._contents:
            if not content.is_valid(error):
                return False
        return True

    def buffer_nbytes(self, names_nbytes):
        names_nbytes[f"node{self._id}-tags"] = self._tags.nbytes()
        names_nbytes[f"node{self._id}-index"] = self._index.nbytes()
        for content in self._contents:
            content.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self._tags.concatenate(buffers[f"node{self._id}-tags"])
        self._index.concatenate(buffers[f"node{self._id}-index"])
        for content in self._contents:
            content.to_buffers(buffers)

    def form(self):
        params = "" if self._parameters == "" else f", parameters: {self._parameters}"
        contents = ", ".join(content.form() for content in self._contents)
        return f'{{"class": "UnionArray", "tags": "{self._tags.index_form()}", "index": "{self._index.index_form()}", "contents": [{contents}], "form_key": "node{self._id}"{params}}}'
