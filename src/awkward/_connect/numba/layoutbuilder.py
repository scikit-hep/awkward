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

    def snapshot(self, *, highlevel=True, behavior=None) -> ArrayLike:
        return self._content.snapshot()

    def _type(self, typestrs):
        raise NotImplementedError


@final
class NumpyBuilder(LayoutBuilder):
    def __init__(self, dtype, *, parameters=None, initial=1024, resize=8.0):
        self._dtype = dtype
        self._data = GrowableBuffer(dtype=dtype, initial=initial, resize=resize)
        self._parameters = parameters
        self._id = 0

    @classmethod
    def _from_buffer(cls, data):
        out = cls.__new__(cls)
        out._dtype = data.dtype
        out._data = data  # GrowableBuffer(dtype=dtype, initial=initial, resize=resize)
        out._parameters = ""  # FIXME: parameters?
        return out

    def __repr__(self):
        return f"<NumpyBuilder of {self.dtype!r} with {self.length} items>"

    def _type(self, typestrs):
        ...

    @property
    def dtype(self):
        return self._dtype

    @property
    def length(self):
        return len(self._data)

    def __len__(self):
        return self.length

    def append(self, x):
        self._data.append(x)

    def extend(self, data):
        self._data.extend(data)

    def parameters(self):
        return self._parameters

    def set_id(self, id: int):
        self._id = id
        id += 1

    def clear(self):
        self._data.clear()

    def is_valid(self, error: str):
        return True

    def snapshot(self) -> ArrayLike:
        """
        Converts the currently accumulated data into an #ak.Array.
        """
        # FIXME: Yes, no numba
        return ak.from_buffers(
            self.form(), self.length, {"node0-data": self._data.snapshot()}
        )

    def form(self):
        # FIXME: no numba
        params = ""
        if self._parameters is not None:
            params = (
                "" if self._parameters == "" else f", parameters: {self._parameters}"
            )
        return f'{{"class": "NumpyArray", "primitive": "{ak.types.numpytype.dtype_to_primitive(self._data.dtype)}", "form_key": "node0"{params}}}'


class NumpyBuilderType(numba.types.Type):
    def __init__(self, dtype):
        super().__init__(name=f"ak.NumpyBuilder({dtype})")
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


@numba.extending.typeof_impl.register(NumpyBuilder)
def typeof_NumpyBuilder(val, c):
    return NumpyBuilderType(numba.from_dtype(val.dtype))


@numba.extending.register_model(NumpyBuilderType)
class NumpyBuilderModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", fe_type.data),
        ]
        super().__init__(dmm, fe_type, members)


for member in ("data",):
    numba.extending.make_attribute_wrapper(NumpyBuilderType, member, "_" + member)


@numba.extending.overload_attribute(NumpyBuilderType, "dtype")
def NumpyBuilderType_dtype(builder):
    def getter(builder):
        return builder.dtype

    return getter


@numba.extending.unbox(NumpyBuilderType)
def NumpyBuilderType_unbox(typ, obj, c):
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


@numba.extending.box(NumpyBuilderType)
def NumpyBuilderType_box(typ, val, c):
    # get PyObject of the NumpyBuilder class
    NumpyBuilder_obj = c.pyapi.unserialize(c.pyapi.serialize_object(NumpyBuilder))
    from_data_obj = c.pyapi.object_getattr_string(NumpyBuilder_obj, "_from_buffer")

    builder = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    data_obj = c.pyapi.from_native_value(typ.data, builder.data, c.env_manager)

    out = c.pyapi.call_function_objargs(from_data_obj, (data_obj,))

    # decref PyObjects
    c.pyapi.decref(NumpyBuilder_obj)
    c.pyapi.decref(from_data_obj)

    c.pyapi.decref(data_obj)

    return out


def _from_buffer():
    ...


@numba.extending.type_callable(_from_buffer)
def NumpyBuilder_from_buffer_typer(context):
    def typer(data):
        if isinstance(data, GrowableBufferType) and isinstance(
            data.dtype, numba.types.Array
        ):
            return NumpyBuilderType(data.dtype.dtype)

    return typer


@numba.extending.lower_builtin(_from_buffer, GrowableBufferType)
def NumpyBuilder_from_buffer_impl(context, builder, sig, args):
    out = numba.core.cgutils.create_struct_proxy(sig.return_type)(context, builder)
    out.data = args

    if context.enable_nrt:
        context.nrt.incref(builder, sig.args[0], args[0])

    return out._getvalue()


@numba.extending.overload(NumpyBuilder)
def NumpyBuilder_ctor(dtype, parameters=None, initial=1024, resize=8.0):
    def ctor_impl(dtype, parameters=None, initial=1024, resize=8.0):
        data = GrowableBuffer(dtype, initial, resize)
        return _from_buffer(data)

    return ctor_impl


@numba.extending.overload_method(NumpyBuilderType, "_length_get", inline="always")
def NumpyBuilder_length(builder):
    def getter(builder):
        return builder._data._length_pos[0]

    return getter


@numba.extending.overload(len)
def NumpyBuilderType_len(builder):
    if isinstance(builder, NumpyBuilderType):

        def len_impl(builder):
            return builder._length_get()

        return len_impl


@numba.extending.overload_method(NumpyBuilderType, "append")
def NumpyBuilder_append(builder, datum):
    def append(builder, datum):
        builder._data.append(datum)

    return append


@numba.extending.overload_method(NumpyBuilderType, "extend")
def NumpyBuilder_extend(builder, data):
    def extend(builder, data):
        builder._data.extend(data)

    return extend


@numba.extending.overload_method(NumpyBuilderType, "snapshot")
def NumpyBuilder_snapshot(builder):
    def snapshot(builder):
        return builder._data.snapshot()

    return snapshot


@final
class EmptyBuilder(LayoutBuilder):
    def __init__(self, *, parameters=None):
        self._parameters = parameters
        self._id = 0

    def __repr__(self):
        return f"<EmptyBuilder with {self.length} items>"

    def parameters(self):
        return self._parameters

    def set_id(self, id: int):
        self._id = id
        id += 1

    def clear(self):
        pass

    @property
    def length(self):
        return 0

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
        return ak.from_buffers(self.form(), self.length, {})


class EmptyBuilderType(numba.types.Type):
    def __init__(self):
        super().__init__(name="ak.EmptyBuilder()")

    @property
    def parameters(self):
        return numba.types.StringLiteral

    @property
    def length(self):
        return numba.types.float64


@numba.extending.typeof_impl.register(EmptyBuilder)
def typeof_EmptyBuilder(val, c):
    return EmptyBuilderType()


@numba.extending.register_model(EmptyBuilderType)
class EmptyBuilderModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [])


@numba.extending.unbox(EmptyBuilderType)
def EmptyBuilderType_unbox(typ, obj, c):
    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@final
class ListOffsetBuilder(LayoutBuilder):
    def __init__(self, dtype, content, *, parameters=None, initial=1024, resize=8.0):
        self._offsets = GrowableBuffer(dtype=dtype, initial=initial, resize=resize)
        self._offsets.append(0)
        self._content = content
        self._parameters = parameters
        self._id = 0

    def __repr__(self):
        return f"<ListOffsetBuilder of {self._content!r} with {self.length} items>"

    def content(self):
        return self._content

    def begin_list(self):
        return self._content

    def end_list(self):
        self._offsets.append(self._content.length())

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
    def length(self):
        return self._offsets.length() - 1

    def is_valid(self, error: str):
        if self._content.length() != self._offsets.last():
            error = f"ListOffset node{self._id} has content length {self._content.length()} but last offset {self._offsets.last()}"
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
        return f'{{"class": "ListOffsetArray", "offsets": "{self._offsets.index_form()}", "content": {self._content.form()}, "form_key": "node{self._id}"{params}}}'


@final
class ListBuilder(LayoutBuilder):
    def __init__(self, PRIMITIVE, content, parameters):
        self._starts = GrowableBuffer(PRIMITIVE)
        self._stops = GrowableBuffer(PRIMITIVE)
        self._content = content
        self._parameters = parameters
        self._id = 0

    def content(self):
        return self._content

    def begin_list(self):
        self._starts.append(self._content.length())
        return self._content

    def end_list(self):
        self._stops.append(self._content.length())

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

    def length(self):
        return self._starts.length()

    def is_valid(self, error: str):
        if self._starts.length() != self._stops.length():
            error = f"List node{self._id} has starts length {self._starts.length()} but stops length {self._stops.length()}"
        elif self._stops.length() > 0 and self._content.length() != self._stops.last():
            error = f"List node{self._id} has content length {self._content.length()} but last stops {self._stops.last()}"
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


@final
class RegularBuilder(LayoutBuilder):
    def __init__(self, content, size, parameters):
        self._length = 0
        self._content = content
        self.size_ = size
        self._parameters = parameters
        self._id = 0

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

    def is_valid(self, error: str):
        if self._content.length() != self._length * self.size_:
            error = f"Regular node{self._id} has content length {self._content.length()}, but length {self._length} and size {self.size_}"
            return False
        else:
            return self._content.is_valid(error)

    def buffer_nbytes(self, names_nbytes):
        self._content.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self._content.to_buffers(buffers)

    def form(self):
        params = "" if self._parameters == "" else f", parameters: {self._parameters}"
        return f'{{"class": "RegularArray", "size": {self.size_}, "content": {self._content.form()}, "form_key": "node{self._id}"{params}}}'


@final
class IndexedBuilder(LayoutBuilder):
    def __init__(self, PRIMITIVE, content, parameters):
        self.last_valid_ = -1
        self.index_ = GrowableBuffer(PRIMITIVE)
        self._content = content
        self._parameters = parameters
        self._id = 0

    def content(self):
        return self._content

    def append_index(self):
        self.last_valid_ = self._content.length()
        self.index_.append(self.last_valid_)
        return self._content

    def extend_index(self, size):
        start = self._content.length()
        stop = start + size
        self.last_valid_ = stop - 1
        self.index_.extend(list(range(start, stop)), size)
        return self._content

    def parameters(self):
        return self._parameters

    def set_id(self, id: int):
        self._id = id
        id += 1
        self._content.set_id(id)

    def clear(self):
        self.last_valid_ = -1
        self.index_.clear()
        self._content.clear()

    def length(self):
        return self.index_.length()

    def is_valid(self, error: str):
        if self._content.length() != self.index_.length():
            error = f"Indexed node{self._id} has content length {self._content.length()} but index length {self.index_.length()}"
            return False
        elif self._content.length() != self.last_valid_ + 1:
            error = f"Indexed node{self._id} has content length {self._content.length()} but last valid index is {self.last_valid_}"
        else:
            return self._content.is_valid(error)

    def buffer_nbytes(self, names_nbytes):
        names_nbytes[f"node{self._id}-index"] = self.index_.nbytes()
        self._content.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self.index_.concatenate(buffers[f"node{self._id}-index"])
        self._content.to_buffers(buffers)

    def form(self):
        params = "" if self._parameters == "" else f", parameters: {self._parameters}"
        return f'{{"class": "IndexedArray", "index": "{self.index_.index_form()}", "content": {self._content.form()}, "form_key": "node{self._id}"{params}}}'


@final
class IndexedOptionBuilder(LayoutBuilder):
    def __init__(self, PRIMITIVE, content, parameters):
        self.last_valid_ = -1
        self.index_ = GrowableBuffer(PRIMITIVE)
        self._content = content
        self._parameters = parameters
        self._id = 0

    def content(self):
        return self._content

    def append_index(self):
        self.last_valid_ = self._content.length()
        self.index_.append(self.last_valid_)
        return self._content

    def extend_index(self, size):
        start = self._content.length()
        stop = start + size
        self.last_valid_ = stop - 1
        self.index_.extend(list(range(start, stop)), size)
        return self._content

    def append_null(self):
        self.index_.append(-1)

    def extend_null(self, size):
        self.index_.extend([-1] * size, size)

    def parameters(self):
        return self._parameters

    def set_id(self, id: int):
        self._id = id
        id += 1
        self._content.set_id(id)

    def clear(self):
        self.last_valid_ = -1
        self.index_.clear()
        self._content.clear()

    def length(self):
        return self.index_.length()

    def is_valid(self, error: str):
        if self._content.length() != self.last_valid_ + 1:
            error = f"Indexed node{self._id} has content length {self._content.length()} but last valid index is {self.last_valid_}"
            return False
        else:
            return self._content.is_valid(error)

    def buffer_nbytes(self, names_nbytes):
        names_nbytes[f"node{self._id}-index"] = self.index_.nbytes()
        self._content.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self.index_.concatenate(buffers[f"node{self._id}-index"])
        self._content.to_buffers(buffers)

    def form(self):
        params = "" if self._parameters == "" else f", parameters: {self._parameters}"
        return f'{{"class": "IndexedOptionArray", "index": "{self.index_.index_form()}", "content": {self._content.form()}, "form_key": "node{self._id}"{params}}}'


@final
class ByteMaskedBuilder(LayoutBuilder):
    def __init__(self, content, valid_when, parameters):
        self._mask = GrowableBuffer("int8")
        self._content = content
        self._valid_when = valid_when
        self._parameters = parameters
        self._id = 0

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
            error = f"ByteMasked node{self._id} has content length {self._content.length()} but mask length {self._stops.length()}"
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


@final
class BitMaskedBuilder(LayoutBuilder):
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


@final
class UnmaskedBuilder(LayoutBuilder):
    def __init__(self, content, parameters):
        self._content = content
        self._parameters = parameters
        self._id = 0

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


class FieldPair:
    def __init__(self, name, content):
        self.name = name
        self.content = content


@final
class RecordBuilder(LayoutBuilder):
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


@final
class TupleBuilder(LayoutBuilder):
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


@final
class EmptyRecordBuilder(LayoutBuilder):
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


@final
class UnionBuilder(LayoutBuilder):
    def __init__(self, PRIMITIVE, contents, parameters):
        self.last_valid_index_ = [-1] * len(contents)
        self.tags_ = GrowableBuffer("int8")
        self.index_ = GrowableBuffer(PRIMITIVE)
        self._contents = contents
        self._parameters = parameters
        self._id = 0

    def append_index(self, tag):
        which_content = self._contents[tag]
        next_index = which_content.length()
        self.last_valid_index_[tag] = next_index
        self.tags_.append(tag)
        self.index_.append(next_index)
        return which_content

    def parameters(self):
        return self._parameters

    def set_id(self, id: int):
        self._id = id
        id += 1
        for content in self._contents:
            content.set_id(id)

    def clear(self):
        for tag in range(len(self.last_valid_index_)):
            self.last_valid_index_[tag] = -1
        self.tags_.clear()
        self.index_.clear()
        for content in self._contents:
            content.clear()

    def length(self):
        return self.tags_.length()

    def is_valid(self, error: str):
        for tag in range(len(self.last_valid_index_)):
            if self._contents[tag].length() != self.last_valid_index_[tag] + 1:
                error = f"Union node{self._id} has content {tag} length {self._contents[tag].length()} but last valid index is {self.last_valid_index_[tag]}"
                return False
        for content in self._contents:
            if not content.is_valid(error):
                return False
        return True

    def buffer_nbytes(self, names_nbytes):
        names_nbytes[f"node{self._id}-tags"] = self.tags_.nbytes()
        names_nbytes[f"node{self._id}-index"] = self.index_.nbytes()
        for content in self._contents:
            content.buffer_nbytes(names_nbytes)

    def to_buffers(self, buffers):
        self.tags_.concatenate(buffers[f"node{self._id}-tags"])
        self.index_.concatenate(buffers[f"node{self._id}-index"])
        for content in self._contents:
            content.to_buffers(buffers)

    def form(self):
        params = "" if self._parameters == "" else f", parameters: {self._parameters}"
        contents = ", ".join(content.form() for content in self._contents)
        return f'{{"class": "UnionArray", "tags": "{self.tags_.index_form()}", "index": "{self.index_.index_form()}", "contents": [{contents}], "form_key": "node{self._id}"{params}}}'
