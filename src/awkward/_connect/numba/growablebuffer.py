# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numba
import numba.core.typing
import numba.core.typing.ctypes_utils
import numpy


class GrowableBuffer:
    def __init__(self, dtype, *, initial=1024, resize=10.0):
        self._panels = [numpy.zeros((initial,), dtype=dtype)]
        self._length = 0
        self._pos = 0
        self._resize = resize

    @classmethod
    def _from_data(cls, panels, length, pos, resize):
        out = cls.__new__(cls)
        out._panels = panels
        out._length = length
        out._pos = pos
        out._resize = resize
        return out

    @property
    def _numba_panels(self):
        return numba.typed.List(self._panels)

    @property
    def dtype(self):
        return self._panels[0].dtype

    def __repr__(self):
        return f"<GrowableBuffer({self.dtype!r}) len {self._length}>"

    def __len__(self):
        return self._length

    def append(self, datum):
        if self._pos == len(self._panels[-1]):
            self._add_panel()

        self._panels[-1][self._pos] = datum
        self._pos += 1
        self._length += 1

    def extend(self, data):
        panel_index = len(self._panels) - 1
        pos = self._pos

        available = len(self._panels[-1]) - self._pos
        while len(data) > available:
            self._add_panel()
            available += len(self._panels[-1])

        remaining = len(data)
        while remaining > 0:
            panel = self._panels[panel_index]
            available_in_panel = len(panel) - pos
            to_write = min(remaining, available_in_panel)

            start = len(data) - remaining
            panel[pos : pos + to_write] = data[start : start + to_write]

            if panel_index == len(self._panels) - 1:
                self._pos += to_write
            remaining -= to_write
            pos = 0
            panel_index += 1

        self._length += len(data)

    def _add_panel(self):
        panel_length = len(self._panels[-1])
        if len(self._panels) == 1:
            # only resize the first time, and by a large factor (C++ should do this, too!)
            panel_length = int(numpy.ceil(panel_length * self._resize))

        self._panels.append(numpy.zeros((panel_length,), dtype=self.dtype))
        self._pos = 0

    def snapshot(self):
        out = numpy.zeros((self._length,), dtype=self.dtype)

        start = 0
        stop = 0
        for panel in self._panels[:-1]:  # full panels, not including the last
            stop += len(panel)
            out[start:stop] = panel
            start = stop

        stop += self._pos
        out[start:stop] = self._panels[-1][: self._pos]

        return out


class GrowableBufferType(numba.types.Type):
    def __init__(self, dtype):
        super().__init__(name=f"ak.GrowableBuffer({dtype})")
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def arraytype(self):
        return numba.types.Array(self.dtype, 1, "C")

    @property
    def listtype(self):
        return numba.types.ListType(self.arraytype)


@numba.extending.typeof_impl.register(GrowableBuffer)
def typeof_GrowableBuffer(val, c):
    return GrowableBufferType(numba.from_dtype(val.dtype))


@numba.extending.register_model(GrowableBufferType)
class GrowableBufferModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("panels", fe_type.listtype),
#            ("last", fe_type.arraytype),
            ("length", numba.types.intp),
            ("pos", numba.types.intp),
            ("resize", numba.types.float64),
        ]
        super().__init__(dmm, fe_type, members)


for member in ("panels", "length", "pos", "resize"):
    numba.extending.make_attribute_wrapper(GrowableBufferType, member, "_" + member)


@numba.extending.overload_attribute(GrowableBufferType, "dtype")
def GrowableBufferType_dtype(growablebuffer):
    def getter(growablebuffer):
        return growablebuffer._panel[0].dtype
    return getter


def _get_last(lst):
    return lst[-1]


@numba.extending.unbox(GrowableBufferType)
def GrowableBufferType_unbox(typ, obj, c):
    # get PyObjects
    panels_obj = c.pyapi.object_getattr_string(obj, "_numba_panels")
    length_obj = c.pyapi.object_getattr_string(obj, "_length")
    pos_obj = c.pyapi.object_getattr_string(obj, "_pos")
    resize_obj = c.pyapi.object_getattr_string(obj, "_resize")

    # lower _get_last and use it to extract the last array from panels
    panels_val = c.pyapi.to_native_value(typ.listtype, panels_obj).value
    args = (panels_val,)
    sig = typ.arraytype(typ.listtype)
    cres = c.context.compile_subroutine(c.builder, _get_last, sig)
    last_val = c.context.call_internal_no_propagate(c.builder, cres.fndesc, sig, args)[1]


    # fill the lowered model
    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out.panels = panels_val
    # out.last = last_val
    out.length = c.pyapi.number_as_ssize_t(length_obj)
    out.pos = c.pyapi.number_as_ssize_t(pos_obj)
    out.resize = c.pyapi.float_as_double(resize_obj)

    # decref PyObjects
    c.pyapi.decref(panels_obj)
    c.pyapi.decref(length_obj)
    c.pyapi.decref(pos_obj)
    c.pyapi.decref(resize_obj)

    # if c.context.enable_nrt:
    #     c.context.nrt.decref(c.builder, typ.arraytype, last_val)

    # return it or the exception
    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)


@numba.extending.box(GrowableBufferType)
def GrowableBufferType_box(typ, val, c):
    print("compile-time")
    numba.core.cgutils.printf(c.builder, "runtime\n")

    # get PyObject of the GrowableBuffer class and _from_data constructor
    GrowableBuffer_obj = c.pyapi.unserialize(c.pyapi.serialize_object(GrowableBuffer))
    from_data_obj = c.pyapi.object_getattr_string(GrowableBuffer_obj, "_from_data")

    growablebuffer = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    panels_obj = c.pyapi.from_native_value(typ.listtype, growablebuffer.panels, c.env_manager)
    length_obj = c.pyapi.long_from_ssize_t(growablebuffer.length)
    pos_obj = c.pyapi.long_from_ssize_t(growablebuffer.pos)
    resize_obj = c.pyapi.float_from_double(growablebuffer.resize)

    out = c.pyapi.call_function_objargs(from_data_obj, (panels_obj, length_obj, pos_obj, resize_obj))

    # decref PyObjects
    c.pyapi.decref(GrowableBuffer_obj)
    c.pyapi.decref(from_data_obj)

    return out
