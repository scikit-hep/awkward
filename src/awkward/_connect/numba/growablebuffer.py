# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import math

import numba
import numba.core.typing.npydecl
import numpy

from awkward.numba import GrowableBuffer


class GrowableBufferType(numba.types.Type):
    def __init__(self, dtype):
        super().__init__(name=f"ak.GrowableBuffer({dtype})")
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def panels(self):
        return numba.types.ListType(numba.types.Array(self.dtype, 1, "C"))

    @property
    def length_pos(self):
        return numba.types.Array(numba.types.int64, 1, "C")

    @property
    def resize(self):
        return numba.types.float64


@numba.extending.typeof_impl.register(GrowableBuffer)
def typeof_GrowableBuffer(val, c):
    return GrowableBufferType(numba.from_dtype(val.dtype))


@numba.extending.register_model(GrowableBufferType)
class GrowableBufferModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("panels", fe_type.panels),
            ("length_pos", fe_type.length_pos),
            ("resize", fe_type.resize),
        ]
        super().__init__(dmm, fe_type, members)


for member in ("panels", "length_pos", "resize"):
    numba.extending.make_attribute_wrapper(GrowableBufferType, member, "_" + member)


@numba.extending.overload_attribute(GrowableBufferType, "dtype")
def GrowableBufferType_dtype(growablebuffer):
    def getter(growablebuffer):
        return growablebuffer._panels[0].dtype

    return getter


@numba.extending.unbox(GrowableBufferType)
def GrowableBufferType_unbox(typ, obj, c):
    # get PyObjects
    panels_obj = c.pyapi.object_getattr_string(obj, "_panels")
    length_pos_obj = c.pyapi.object_getattr_string(obj, "_length_pos")
    resize_obj = c.pyapi.object_getattr_string(obj, "_resize")

    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out.panels = c.pyapi.to_native_value(typ.panels, panels_obj).value
    out.length_pos = c.pyapi.to_native_value(typ.length_pos, length_pos_obj).value
    out.resize = c.pyapi.to_native_value(typ.resize, resize_obj).value

    # decref PyObjects
    c.pyapi.decref(panels_obj)
    c.pyapi.decref(length_pos_obj)
    c.pyapi.decref(resize_obj)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(GrowableBufferType)
def GrowableBufferType_box(typ, val, c):
    # get PyObject of the GrowableBuffer class and _from_data constructor
    GrowableBuffer_obj = c.pyapi.unserialize(c.pyapi.serialize_object(GrowableBuffer))
    from_data_obj = c.pyapi.object_getattr_string(GrowableBuffer_obj, "_from_data")

    growablebuffer = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, value=val
    )
    panels_obj = c.pyapi.from_native_value(
        typ.panels, growablebuffer.panels, c.env_manager
    )
    length_pos_obj = c.pyapi.from_native_value(
        typ.length_pos, growablebuffer.length_pos, c.env_manager
    )
    resize_obj = c.pyapi.from_native_value(
        typ.resize, growablebuffer.resize, c.env_manager
    )

    out = c.pyapi.call_function_objargs(
        from_data_obj, (panels_obj, length_pos_obj, resize_obj)
    )

    # decref PyObjects
    c.pyapi.decref(GrowableBuffer_obj)
    c.pyapi.decref(from_data_obj)

    c.pyapi.decref(panels_obj)
    c.pyapi.decref(length_pos_obj)
    c.pyapi.decref(resize_obj)

    return out


def _from_data():
    raise RuntimeError("_from_data Python function is only implemented in Numba")


@numba.extending.type_callable(_from_data)
def GrowableBuffer_from_data_typer(context):
    def typer(panels, length_pos, resize):
        if (
            isinstance(panels, numba.types.ListType)
            and isinstance(panels.dtype, numba.types.Array)
            and isinstance(length_pos, numba.types.Array)
            and isinstance(resize, numba.types.Float)
        ):
            return GrowableBufferType(panels.dtype.dtype)

    return typer


@numba.extending.lower_builtin(
    _from_data, numba.types.ListType, numba.types.Array, numba.types.Float
)
def GrowableBuffer_from_data_impl(context, builder, sig, args):
    out = numba.core.cgutils.create_struct_proxy(sig.return_type)(context, builder)
    out.panels, out.length_pos, out.resize = args

    if context.enable_nrt:
        context.nrt.incref(builder, sig.args[0], args[0])
        context.nrt.incref(builder, sig.args[1], args[1])

    return out._getvalue()


@numba.extending.overload(GrowableBuffer)
def GrowableBuffer_ctor(dtype, initial=1024, resize=8.0):
    if isinstance(dtype, numba.types.StringLiteral):
        dt = numpy.dtype(dtype.literal_value)

    elif isinstance(dtype, numba.types.DTypeSpec):
        dt = numba.core.typing.npydecl.parse_dtype(dtype)

    else:
        return

    def ctor_impl(dtype, initial=1024, resize=8.0):
        panels = numba.typed.List([numpy.empty((initial,), dt)])
        length_pos = numpy.zeros((2,), dtype=numpy.int64)
        return _from_data(panels, length_pos, resize)

    return ctor_impl


@numba.extending.overload_method(GrowableBufferType, "_length_get", inline="always")
def GrowableBuffer_length_get(growablebuffer):
    def getter(growablebuffer):
        return growablebuffer._length_pos[0]

    return getter


@numba.extending.overload_method(GrowableBufferType, "_pos_get", inline="always")
def GrowableBuffer_pos_get(growablebuffer):
    def getter(growablebuffer):
        return growablebuffer._length_pos[1]

    return getter


@numba.extending.overload_method(GrowableBufferType, "_length_set", inline="always")
def GrowableBuffer_length_set(growablebuffer, value):
    def setter(growablebuffer, value):
        growablebuffer._length_pos[0] = value

    return setter


@numba.extending.overload_method(GrowableBufferType, "_pos_set", inline="always")
def GrowableBuffer_pos_set(growablebuffer, value):
    def setter(growablebuffer, value):
        growablebuffer._length_pos[1] = value

    return setter


@numba.extending.overload_method(GrowableBufferType, "_length_inc", inline="always")
def GrowableBuffer_length_inc(growablebuffer, value):
    def inccer(growablebuffer, value):
        growablebuffer._length_pos[0] += value

    return inccer


@numba.extending.overload_method(GrowableBufferType, "_pos_inc", inline="always")
def GrowableBuffer_pos_inc(growablebuffer, value):
    def inccer(growablebuffer, value):
        growablebuffer._length_pos[1] += value

    return inccer


@numba.extending.overload(len)
def GrowableBufferType_len(growablebuffer):
    if isinstance(growablebuffer, GrowableBufferType):

        def len_impl(growablebuffer):
            return growablebuffer._length_get()

        return len_impl


@numba.extending.overload_method(GrowableBufferType, "_add_panel")
def GrowableBuffer_add_panel(growablebuffer):
    def add_panel(growablebuffer):
        first_panel = growablebuffer._panels[0]
        panel_length = int(math.ceil(len(first_panel) * growablebuffer._resize))

        growablebuffer._panels.append(numpy.empty((panel_length,), first_panel.dtype))
        growablebuffer._pos_set(0)

    return add_panel


@numba.extending.overload_method(GrowableBufferType, "append")
def GrowableBuffer_append(growablebuffer, datum):
    def append(growablebuffer, datum):
        if growablebuffer._pos_get() == len(growablebuffer._panels[-1]):
            growablebuffer._add_panel()

        growablebuffer._panels[-1][growablebuffer._pos_get()] = datum
        growablebuffer._pos_inc(1)
        growablebuffer._length_inc(1)

    return append


@numba.extending.overload_method(GrowableBufferType, "extend")
def GrowableBuffer_extend(growablebuffer, data):
    def extend(growablebuffer, data):
        panel_index = len(growablebuffer._panels) - 1
        pos = growablebuffer._pos_get()

        available = len(growablebuffer._panels[-1]) - pos
        remaining = len(data)

        if remaining > available:
            panel_length = int(
                math.ceil(len(growablebuffer._panels[0]) * growablebuffer._resize)
            )

            growablebuffer._panels.append(
                numpy.empty((max(remaining, panel_length),), dtype=growablebuffer.dtype)
            )
            growablebuffer._pos_set(0)
            available += len(growablebuffer._panels[-1])

        while remaining > 0:
            panel = growablebuffer._panels[panel_index]
            available_in_panel = len(panel) - pos
            to_write = min(remaining, available_in_panel)

            start = len(data) - remaining
            panel[pos : pos + to_write] = data[start : start + to_write]

            if panel_index == len(growablebuffer._panels) - 1:
                growablebuffer._pos_inc(to_write)
            remaining -= to_write
            pos = 0
            panel_index += 1

        growablebuffer._length_inc(len(data))

    return extend


@numba.extending.overload_method(GrowableBufferType, "snapshot")
def GrowableBuffer_snapshot(growablebuffer):
    def snapshot(growablebuffer):
        out = numpy.empty((growablebuffer._length_get(),), growablebuffer.dtype)

        start = 0
        stop = 0
        for panel in growablebuffer._panels[:-1]:  # full panels, not including the last
            stop += len(panel)
            out[start:stop] = panel
            start = stop

        stop += growablebuffer._pos_get()
        out[start:stop] = growablebuffer._panels[-1][: growablebuffer._pos_get()]

        return out

    return snapshot
