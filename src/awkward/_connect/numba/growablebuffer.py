# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numba
import numba.core.typing
import numba.core.typing.ctypes_utils
import numpy


class GrowableBuffer:
    def __init__(self, dtype, *, initial=1024, resize=10.0):
        self._panels = [numpy.zeros((initial,), dtype=dtype)]
        self._length = 0
        self._last_panel_length = 0
        self._resize = resize

    @property
    def dtype(self):
        return self._panels[0].dtype

    def __repr__(self):
        return f"<GrowableBuffer({self.dtype!r}) len {self._length}>"

    def __len__(self):
        return self._length

    def append(self, datum):
        if self._last_panel_length == len(self._panels[-1]):
            self._add_panel()

        self._panels[-1][self._last_panel_length] = datum
        self._last_panel_length += 1
        self._length += 1

    def extend(self, data):
        panel_index = len(self._panels) - 1
        pos = self._last_panel_length

        available = len(self._panels[-1]) - self._last_panel_length
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
                self._last_panel_length += to_write
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
        self._last_panel_length = 0

    def snapshot(self):
        out = numpy.zeros((self._length,), dtype=self.dtype)

        start = 0
        stop = 0
        for panel in self._panels[:-1]:  # full panels, not including the last
            stop += len(panel)
            out[start:stop] = panel
            start = stop

        stop += self._last_panel_length
        out[start:stop] = self._panels[-1][: self._last_panel_length]

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
        return numba.types.List(self.arraytype)


@numba.extending.typeof_impl.register(GrowableBuffer)
def typeof_GrowableBuffer(val, c):
    return GrowableBufferType(numba.from_dtype(val.dtype))


@numba.extending.register_model(GrowableBufferType)
class GrowableBufferModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("_panels", fe_type.listtype),
            ("_length", numba.types.int64),
            ("_last_panel_length", numba.types.int64),
            ("_resize", numba.types.float64),
        ]
        super().__init__(dmm, fe_type, members)


for member in ("_panels", "_length", "_last_panel_length", "_resize"):
    numba.extending.make_attribute_wrapper(GrowableBufferType, member, member)


@numba.extending.overload_attribute(GrowableBufferType, "dtype")
def GrowableBufferType_dtype(growablebuffer):
    def getter(growablebuffer):
        return growablebuffer._panel[0].dtype
    return getter


@numba.extending.unbox(GrowableBufferType)
def GrowableBufferType_unbox(typ, obj, c):
    print("compile-time")
    numba.core.cgutils.printf(c.builder, "runtime\n")

    # get PyObjects
    panels_obj = c.pyapi.object_getattr_string(obj, "_panels")
    length_obj = c.pyapi.object_getattr_string(obj, "_length")
    last_panel_length_obj = c.pyapi.object_getattr_string(obj, "_last_panel_length")
    resize_obj = c.pyapi.object_getattr_string(obj, "_resize")

    out = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    out._panels = c.pyapi.to_native_value(typ.listtype, panels_obj).value
    out._length = c.pyapi.to_native_value(numba.types.int64, length_obj).value
    out._last_panel_length = c.pyapi.to_native_value(numba.types.int64, last_panel_length_obj).value
    out._resize = c.pyapi.to_native_value(numba.types.float64, resize_obj).value

    c.pyapi.decref(panels_obj)
    c.pyapi.decref(length_obj)
    c.pyapi.decref(last_panel_length_obj)
    c.pyapi.decref(resize_obj)

    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(out._getvalue(), is_error=is_error)
