# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import math

import numba
import numba.core.typing.npydecl
import numpy

import awkward as ak

from awkward._connect.numba.growablebuffer import GrowableBuffer

class Ref:
    def __init__(self, value):
        self.value = value


class GenericRef(Ref):
    def __init__(self, get, set):
        self.get = get
        self.set = set

    @property
    def value(self):
        return self.get()

    @value.setter
    def value(self, new_value):
        return self.set(new_value)

class NumpyBuilder:
    def __init__(self, dtype, parameters, initial, resize):
        self._data = GrowableBuffer(dtype=dtype, initial=initial, resize=resize)
        self._parameters = parameters
        self.set_id(Ref(0))

    def __repr__(self):
        return f"<Numpy of {self.dtype!r} with {self.length} items>"

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def length(self):
        return len(self._data)

    def append(self, x):
        self._data.append(x)

    def extend(self, data):
        self._data.extend(data)

    def parameters(self):
        return self._parameters

    def set_id(self, id: Ref(int)):
        self._id = id.value
        id.value += 1

    def clear(self):
        self._data.clear()

    def is_valid(self, error: Ref(str)):
        return True

    def snapshot(self):
        return self._data.snapshot()

    # def buffer_nbytes(self, names_nbytes):
    #     names_nbytes[f"node{self._id}-data"] = self._data.nbytes()
    #
    # def to_buffers(self, buffers):
    #     self._data.concatenate(buffers[f"node{self._id}-data"])

    def form(self):
        params = "" if self._parameters == "" else f", parameters: {self._parameters}"
        return f'{{"class": "NumpyArray", "primitive": "{ak.types.numpytype.dtype_to_primitive(self._data.dtype)}", "form_key": "node{self._id}"{params}}}'

#
# class NumpyBuilderType(numba.types.Type):
#     def __init__(self, dtype):
#         super().__init__(name=f"ak.NumpyBuilder({dtype})")
#         self._dtype = dtype
#
#     @property
#     def dtype(self):
#         return self._dtype
#
#
# @numba.extending.typeof_impl.register(NumpyBuilder)
# def typeof_NumpyBuilder(val, c):
#     return NumpyBuilderType(numba.from_dtype(val.dtype))
#
#
# @numba.extending.register_model(NumpyBuilderType)
# class NumpyBuilderModel(numba.extending.models.StructModel):
#     def __init__(self, dmm, fe_type):
#         members = [
#             ("panels", fe_type.panels),
#             ("length_pos", fe_type.length_pos),
#             ("resize", fe_type.resize),
#         ]
#         super().__init__(dmm, fe_type, members)
#
#
# for member in ("panels", "length_pos", "resize"):
#     numba.extending.make_attribute_wrapper(NumpyBuilderType, member, "_" + member)
#
#
# @numba.extending.overload_attribute(NumpyBuilderType, "dtype")
# def NumpyBuilderType_dtype(builder):
#     def getter(builder):
#         return builder.dtype
#
#     return getter
#
#
# @numba.extending.unbox(NumpyBuilderType)
# def NumpyBuilderType_unbox(typ, obj, c):
#     # get PyObjects
#     panels_obj = c.pyapi.object_getattr_string(obj, "_panels")
#     length_pos_obj = c.pyapi.object_getattr_string(obj, "_length_pos")
#     resize_obj = c.pyapi.object_getattr_string(obj, "_resize")
#
#     # fill the lowered model
#     out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
#     out.panels = c.pyapi.to_native_value(typ.panels, panels_obj).value
#     out.length_pos = c.pyapi.to_native_value(typ.length_pos, length_pos_obj).value
#     out.resize = c.pyapi.to_native_value(typ.resize, resize_obj).value
#
#     # decref PyObjects
#     c.pyapi.decref(panels_obj)
#     c.pyapi.decref(length_pos_obj)
#     c.pyapi.decref(resize_obj)
#
#     # return it or the exception
#     is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
#     return numba.extending.NativeValue(out._getvalue(), is_error=is_error)
#
#
# @numba.extending.overload(NumpyBuilder)
# def NumpyBuilder_ctor(dtype):
#     if isinstance(dtype, numba.types.StringLiteral):
#         dt = numpy.dtype(dtype.literal_value)
#
#     elif isinstance(dtype, numba.types.DTypeSpec):
#         dt = numba.core.typing.npydecl.parse_dtype(dtype)
#
#     else:
#         return
#
#     def ctor_impl(dtype, initial=1024, resize=8.0):
#         panels = numba.typed.List([numpy.empty((initial,), dt)])
#         length_pos = numpy.zeros((2,), dtype=numpy.int64)
#         return _from_data(panels, length_pos, resize)
#
#     return ctor_impl
#
#
# @numba.extending.overload_method(NumpyBuilderType, "_length_get", inline="always")
# def NumpyBuilder_length_get(builder):
#     def getter(builder):
#         return builder._length
#
#     return getter
#
#
# @numba.extending.overload_method(NumpyBuilderType, "append")
# def NumpyBuilder_append(builder, datum):
#     def append(builder, datum):
#         if builder._pos_get() == len(builder._panels[-1]):
#             builder._add_panel()
#
#         builder._panels[-1][builder._pos_get()] = datum
#         builder._pos_inc(1)
#         builder._length_inc(1)
#
#     return append
#
#
# @numba.extending.overload_method(NumpyBuilderType, "extend")
# def NumpyBuilder_extend(builder, data):
#     def extend(builder, data):
#         panel_index = len(builder._panels) - 1
#         pos = builder._pos_get()
#
#         available = len(builder._panels[-1]) - pos
#         remaining = len(data)
#
#         if remaining > available:
#             panel_length = int(
#                 math.ceil(len(growablebuffer._panels[0]) * growablebuffer._resize)
#             )
#
#             growablebuffer._panels.append(
#                 numpy.empty((max(remaining, panel_length),), dtype=growablebuffer.dtype)
#             )
#             growablebuffer._pos_set(0)
#             available += len(growablebuffer._panels[-1])
#
#         while remaining > 0:
#             panel = growablebuffer._panels[panel_index]
#             available_in_panel = len(panel) - pos
#             to_write = min(remaining, available_in_panel)
#
#             start = len(data) - remaining
#             panel[pos : pos + to_write] = data[start : start + to_write]
#
#             if panel_index == len(growablebuffer._panels) - 1:
#                 growablebuffer._pos_inc(to_write)
#             remaining -= to_write
#             pos = 0
#             panel_index += 1
#
#         growablebuffer._length_inc(len(data))
#
#     return extend
#
#
# @numba.extending.overload_method(NumpyBuilderType, "snapshot")
# def NumpyBuilder_snapshot(growablebuffer):
#     def snapshot(builder):
#         out = numpy.empty((builder._length_get(),), builder.dtype)
#
#         start = 0
#         stop = 0
#         for panel in builder._panels[:-1]:  # full panels, not including the last
#             stop += len(panel)
#             out[start:stop] = panel
#             start = stop
#
#         stop += builder._pos_get()
#         out[start:stop] = builder._panels[-1][: builder._pos_get()]
#
#         return out
#
#     return snapshot
