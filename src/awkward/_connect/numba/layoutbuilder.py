# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import numba
import numba.core.typing.npydecl

import awkward as ak
from awkward._connect.numba.growablebuffer import GrowableBuffer, GrowableBufferType


class NumpyBuilder:
    def __init__(self, dtype, *, parameters="", initial=1024, resize=8.0):
        self._dtype = dtype
        self._data = GrowableBuffer(dtype=dtype, initial=initial, resize=resize)
        self._parameters = parameters
        self.set_id(0)

    @classmethod
    def _from_data(cls, data):
        out = cls.__new__(cls)
        out._dtype = data.dtype
        out._data = data  # GrowableBuffer(dtype=dtype, initial=initial, resize=resize)
        out._parameters = ""  # FIXME: parameters?
        return out

    def __repr__(self):
        return f"<NumpyBuilder of {self.dtype!r} with {self.length} items>"

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

    def snapshot(self):
        """
        Converts the currently accumulated data into an #ak.Array.
        """
        # FIXME:
        return ak.contents.NumpyArray(self._data.snapshot())

    def form(self):
        # FIXME:
        params = "" if self._parameters == "" else f", parameters: {self._parameters}"
        return f'{{"class": "NumpyArray", "primitive": "{ak.types.numpytype.dtype_to_primitive(self._data.dtype)}", "form_key": "node{self._id}"{params}}}'


class NumpyBuilderType(numba.types.Type):
    def __init__(self, dtype):
        super().__init__(name=f"ak.NumpyBuilder({dtype})")
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def data(self):
        return ak.numba.GrowableBufferType(self._dtype)

    @property
    def length(self):
        return numba.types.float64

    @property
    def parameters(self):
        return numba.types.StringLiteral


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
    from_data_obj = c.pyapi.object_getattr_string(NumpyBuilder_obj, "_from_data")

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


def _from_data():
    ...


@numba.extending.type_callable(_from_data)
def NumpyBuilder_from_data_typer(context):
    def typer(data):
        if isinstance(data, GrowableBufferType) and isinstance(
            data.dtype, numba.types.Array
        ):
            return NumpyBuilderType(data.dtype.dtype)

    return typer


@numba.extending.lower_builtin(_from_data, GrowableBufferType)
def NumpyBuilder_from_data_impl(context, builder, sig, args):
    out = numba.core.cgutils.create_struct_proxy(sig.return_type)(context, builder)
    out.data = args

    if context.enable_nrt:
        context.nrt.incref(builder, sig.args[0], args[0])

    return out._getvalue()


@numba.extending.overload(NumpyBuilder)
def NumpyBuilder_ctor(dtype):
    def ctor_impl(dtype, parameters=None, initial=1024, resize=8.0):
        builder = NumpyBuilder(dtype, parameters, initial, resize)
        return builder

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
