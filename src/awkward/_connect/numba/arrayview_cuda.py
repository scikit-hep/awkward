# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import operator

import numba
import numba.core.typing
import numba.core.typing.ctypes_utils
from numba.core.extending import intrinsic as _intrinsic

intrinsic = _intrinsic(target="cuda")

from numba import cuda, types
from numba.cuda.cudaimpl import registry as cuda_registry

import awkward as ak

np = ak.nplikes.NumpyMetadata.instance()

import _ctypes


def di(obj_id):
    """Inverse of id() function."""
    return _ctypes.PyObj_FromPtr(obj_id)


def pyarrow_cuda_buffer_as_random(size):
    import numpy as np
    import pyarrow as pa
    import pyarrow.cuda as cuda

    ctx = cuda.Context()
    dtype = np.dtype("uint8")
    buf = pa.allocate_buffer(size * dtype.itemsize)
    arr = np.frombuffer(buf, dtype=dtype)
    arr[:] = np.random.randint(low=0, high=255, size=size, dtype=np.uint8)
    cbuf = ctx.new_buffer(buf.size)
    cbuf.copy_from_host(buf, position=0, nbytes=buf.size)
    return cbuf


def pyarrow_cuda_buffer_as_numpy_ndarray(cbuf):
    """Return a copy of CudaBuffer data as a numpy.ndarray."""
    import numpy as np

    dtype = np.dtype("uint8")
    return np.frombuffer(cbuf.copy_to_host(), dtype=dtype)


def pyarrow_cuda_buffer_as_numba_cuda_DeviceNDArray(cbuf):
    """Return numba DeviceNDArray view of a pyarrow.cuda.CudaBuffer."""
    import numpy as np
    from numba.cuda.cudadrv.devicearray import DeviceNDArray

    dtype = np.dtype("uint8")
    return DeviceNDArray(
        (cbuf.size,), (dtype.itemsize,), dtype, gpu_data=cbuf.to_numba()
    )


def pyarrow_cuda_buffer_as_cupy_cuda_MemoryPointer(cbuf):
    """Return cupy.cuda.MemoryPointer view of a pyarrow.cuda.CudaBuffer."""
    import cupy

    # addr = cbuf.context.get_device_address(cbuf.address) # requires arrow>=0.12.1
    addr = cbuf.address
    mem = cupy.cuda.UnownedMemory(addr, cbuf.size, cbuf)
    return cupy.cuda.MemoryPointer(mem, 0)


def pyarrow_cuda_buffer_as_cupy_ndarray(cbuf):
    """Return cupy.ndarray view of a pyarrow.cuda.CudaBuffer."""
    import cupy

    return cupy.ndarray(
        cbuf.size, dtype=cupy.uint8, memptr=cupy_cuda_MemoryPointer(cbuf)
    )


def pyarrow_cuda_buffer_as_xnd_xnd_cuda(cbuf):
    """Return xnd.xnd view of a pyarrow.cuda.CudaBuffer [EXPERIMENTAL]."""
    import pyarrow as pa
    import xnd

    # addr = cbuf.context.get_device_address(cbuf.address) # requires arrow>=0.12.1
    addr = cbuf.address
    # device = cbuf.context.device_number
    buf = pa.foreign_buffer(addr, cbuf.size, cbuf)
    return xnd.xnd.from_buffer(buf)


def cupy_ndarray_as_random(size):
    """Return random cupy.ndarray instance of 8 bit insigned integers."""
    import cupy

    return cupy.random.randint(0, 256, dtype=cupy.uint8, size=size)


def cupy_ndarray_as_numpy_ndarray(cp_arr):
    """Return a copy of CudaBuffer data as a numpy.ndarray."""
    import cupy

    return cupy.asnumpy(cp_arr)


def cupy_ndarray_as_numba_cuda_DeviceNDArray(cp_arr):
    """Return numba DeviceNDArray view of cupy.ndarray."""
    import numba.cuda as nb_cuda

    return nb_cuda.as_cuda_array(cp_arr)


def cupy_ndarray_as_pyarrow_cuda_buffer(cp_arr):
    """Return pyarrow.cuda.CudaBuffer view of cupy.ndarray."""
    import pyarrow.cuda as cuda

    ctx = cuda.Context(cp_arr.data.device.id)
    return ctx.foreign_buffer(cp_arr.data.ptr, cp_arr.nbytes)


def cupy_ndarray_as_cupy_cuda_MemoryPointer(cp_arr):
    """Return cupy.cuda.MemoryPointer view of cupy.ndarray."""
    return cp_arr.data


def numba_cuda_DeviceNDArray_as_random(size):
    import numba.cuda as cuda
    import numpy as np

    arr = np.random.randint(low=0, high=255, size=size, dtype=np.uint8)
    return cuda.to_device(arr)


def numba_cuda_DeviceNDArray_as_numpy_ndarray(nb_arr):
    """Return a copy of numba DeviceNDArray data as a numpy.ndarray."""
    return nb_arr.copy_to_host()


def numba_cuda_DeviceNDArray_as_pyarrow_cuda_buffer(nb_arr):
    """Return pyarrow.cuda.CudaBuffer view of a numba DeviceNDArray."""
    import pyarrow.cuda as cuda

    ctx = cuda.Context()
    if hasattr(ctx, "buffer_from_object"):
        # buffer_from_object is defined in arrow>=0.12.1
        return ctx.buffer_from_object(nb_arr)
    desc = nb_arr.__cuda_array_interface__
    addr = desc["data"][0]
    size = nb_arr.alloc_size
    strides = desc.get("strides")
    assert strides in [(1,), None], repr(strides)
    return ctx.foreign_buffer(addr, size)


def numba_cuda_DeviceNDArray_as_cupy_cuda_MemoryPointer(nb_arr):
    """Return cupy.cuda.MemoryPointer view of a numba DeviceNDArray."""
    import cupy

    addr = nb_arr.device_ctypes_pointer.value
    size = nb_arr.alloc_size
    mem = cupy.cuda.UnownedMemory(addr, size, nb_arr)
    return cupy.cuda.MemoryPointer(mem, 0)


def numba_cuda_DeviceNDArray_as_cupy_ndarray(nb_arr):
    """Return cupy.ndarray view of a numba DeviceNDArray."""
    import cupy

    return cupy.ndarray(
        nb_arr.shape,
        dtype=cupy.uint8,
        strides=nb_arr.strides,
        memptr=numba_cuda_DeviceNDArray_as_cupy_cuda_MemoryPointer(nb_arr),
    )


def numba_cuda_DeviceNDArray_as_xnd_xnd_cuda(nb_arr):
    """Return xnd.xnd view of a numba DeviceNDArray."""
    cbuf = pyarrow_cuda_buffer(nb_arr)
    # DERIVED
    return pyarrow_cuda_buffer_as.xnd_xnd_cuda(cbuf)


def numba_cuda_DeviceNDArray_as_cudf_Series(nb_arr):
    """Return cudf.Series view of a numba DeviceNDArray."""
    import cudf

    return cudf.Series(nb_arr)


#####################################################################


def code_to_function(code, function_name, externals=None, debug=True):
    print("arrayview_cuda.py line 21: code_to_function")
    if debug:
        print("################### " + function_name)  # noqa: T201
        print(code)  # noqa: T201
    namespace = {} if externals is None else dict(externals)
    exec(code, namespace)
    return namespace[function_name]


def tonumbatype(form):
    print("arrayview_cuda.py line 31: tonumbatype")
    if isinstance(form, ak.forms.EmptyForm):
        return tonumbatype(form.toNumpyForm(np.dtype(np.float64)))

    elif isinstance(form, ak.forms.NumpyForm):
        if len(form.inner_shape) == 0:
            return ak._connect.numba.layout.NumpyArrayType.from_form(form)
        else:
            return tonumbatype(form.toRegularForm())

    elif isinstance(form, ak.forms.RegularForm):
        return ak._connect.numba.layout.RegularArrayType.from_form(form)

    elif isinstance(form, (ak.forms.ListForm, ak.forms.ListOffsetForm)):
        return ak._connect.numba.layout.ListArrayType.from_form(form)

    elif isinstance(form, ak.forms.IndexedForm):
        return ak._connect.numba.layout.IndexedArrayType.from_form(form)

    elif isinstance(form, ak.forms.IndexedOptionForm):
        return ak._connect.numba.layout.IndexedOptionArrayType.from_form(form)

    elif isinstance(form, ak.forms.ByteMaskedForm):
        return ak._connect.numba.layout.ByteMaskedArrayType.from_form(form)

    elif isinstance(form, ak.forms.BitMaskedForm):
        return ak._connect.numba.layout.BitMaskedArrayType.from_form(form)

    elif isinstance(form, ak.forms.UnmaskedForm):
        return ak._connect.numba.layout.UnmaskedArrayType.from_form(form)

    elif isinstance(form, ak.forms.RecordForm):
        return ak._connect.numba.layout.RecordArrayType.from_form(form)

    elif isinstance(form, ak.forms.UnionForm):
        return ak._connect.numba.layout.UnionArrayType.from_form(form)

    else:
        raise AssertionError(f"unrecognized Form: {type(form)}")


########## Lookup


@numba.extending.typeof_impl.register(ak._lookup.Lookup)
def typeof_Lookup(obj, c):
    print("arrayview_cuda.py line 77: typeof_Lookup")
    return LookupType()


class LookupType(numba.types.Type):
    arraytype = numba.types.Array(numba.intp, 1, "C")

    def __init__(self):
        print("arrayview_cuda.py line 85: LookupType::__init__")
        super().__init__(name="ak.LookupType()")


@numba.extending.register_model(LookupType)
class LookupModel(numba.core.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        print("arrayview_cuda.py line 92: LookupModel::__init__")
        members = [("arrayptrs", fe_type.arraytype)]
        super().__init__(dmm, fe_type, members)


@numba.extending.unbox(LookupType)
def unbox_Lookup(lookuptype, lookupobj, c):
    print("arrayview_cuda.py line 99: unbox_Lookup", lookuptype, lookupobj, c)
    arrayptrs_obj = c.pyapi.object_getattr_string(lookupobj, "arrayptrs")

    proxyout = c.context.make_helper(c.builder, lookuptype)
    proxyout.arrayptrs = c.pyapi.to_native_value(
        lookuptype.arraytype, arrayptrs_obj
    ).value

    c.pyapi.decref(arrayptrs_obj)

    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)


########## ArrayView


class ArrayView:
    @classmethod
    def fromarray(cls, array):
        print("arrayview_cuda.py line 119: ArrayView::fromarray")
        behavior = ak._util.behavior_of(array)
        layout = ak.operations.to_layout(
            array,
            allow_record=False,
            allow_other=False,
            numpytype=(np.number, np.bool_, np.datetime64, np.timedelta64),
        )

        return ArrayView(
            tonumbatype(layout.form),
            behavior,
            ak._lookup.Lookup(layout),
            0,
            0,
            len(layout),
            (),
        )

    def __init__(self, type, behavior, lookup, pos, start, stop, fields):
        print("arrayview_cuda.py line 139: ArrayView::__init__")
        self.type = type
        self.behavior = behavior
        self.lookup = lookup
        self.pos = pos
        self.start = start
        self.stop = stop
        self.fields = fields
        print(
            "arrayview_cuda.py line 147: ArrayView __init__",
            self,
            self.lookup,
            self.pos,
            self.start,
            self.stop,
        )

    def toarray(self):
        print("arrayview_cuda.py line 150: ArrayView::toarray")
        layout = self.type.tolayout(self.lookup, self.pos, self.fields)
        sliced = layout._getitem_range(slice(self.start, self.stop))
        return ak._util.wrap(sliced, self.behavior)


@numba.extending.typeof_impl.register(ArrayView)
def typeof_ArrayView(obj, c):
    print("arrayview_cuda.py line 158: typeof_ArrayView", obj, c)
    return ArrayViewType(obj.type, obj.behavior, obj.fields)


def wrap(type, viewtype, fields):
    print("arrayview_cuda.py line 163: wrap")
    if fields is None:
        return ArrayViewType(type, viewtype.behavior, viewtype.fields)
    else:
        return ArrayViewType(type, viewtype.behavior, fields)


def repr_behavior(behavior):
    print("arrayview_cuda.py line 171: repr_behavior")
    return repr(behavior)


class ArrayViewType(numba.types.IterableType, numba.types.Sized):
    def __init__(self, type, behavior, fields):
        print("arrayview_cuda.py line 177: ArrayViewType::__init__")
        super().__init__(
            name="ak.ArrayView({}, {}, {})".format(
                type.name, repr_behavior(behavior), repr(fields)
            )
        )
        self.type = type
        self.behavior = behavior
        self.fields = fields

    @property
    def iterator_type(self):
        print("arrayview_cuda.py line 189: iterator_type")
        return IteratorType(self)


@numba.extending.register_model(ArrayViewType)
class ArrayViewModel(numba.core.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        print("arrayview_cuda.py line 196: ArrayViewModel::__init__")
        members = [
            ("pos", numba.intp),
            ("start", numba.intp),
            ("stop", numba.intp),
            ("arrayptrs", numba.types.CPointer(numba.intp)),
            ("pylookup", numba.types.pyobject),
        ]
        super().__init__(dmm, fe_type, members)


@numba.core.imputils.lower_constant(ArrayViewType)
def lower_const_Array(context, builder, viewtype, array):
    print("arrayview_cuda.py line 209: lower_const_Array for CPU")
    return lower_const_view(context, builder, viewtype, array._numbaview)


@cuda_registry.lower_constant(ArrayViewType)
def lower_const_Array(context, builder, viewtype, array):
    print("arrayview_cuda.py line 216: lower_const_Array for CUDA")
    return lower_const_view(context, builder, viewtype, array._numbaview)


def lower_const_view(context, builder, viewtype, view):
    print("arrayview_cuda.py line 214: lower_const_view")
    pos = view.pos
    start = view.start
    stop = view.stop
    lookup = view.lookup
    arrayptrs = lookup.arrayptrs

    arrayptrs_val = context.make_constant_array(
        builder, numba.typeof(arrayptrs), arrayptrs
    )

    proxyout = context.make_helper(builder, viewtype)
    proxyout.pos = context.get_constant(numba.intp, pos)
    proxyout.start = context.get_constant(numba.intp, start)
    proxyout.stop = context.get_constant(numba.intp, stop)
    proxyout.arrayptrs = context.make_helper(
        builder, numba.typeof(arrayptrs), arrayptrs_val
    ).data
    proxyout.pylookup = context.add_dynamic_addr(
        builder, id(lookup), info=str(type(lookup))
    )

    return proxyout._getvalue()


@numba.extending.unbox(ArrayViewType)
def unbox_Array(viewtype, arrayobj, c):
    print("arrayview_cuda.py line 241: unbox_Array")
    view_obj = c.pyapi.object_getattr_string(arrayobj, "_numbaview")
    out = unbox_ArrayView(viewtype, view_obj, c)
    c.pyapi.decref(view_obj)
    return out


def unbox_ArrayView(viewtype, view_obj, c):
    print("arrayview_cuda.py line 249: unbox_ArrayView")
    pos_obj = c.pyapi.object_getattr_string(view_obj, "pos")
    start_obj = c.pyapi.object_getattr_string(view_obj, "start")
    stop_obj = c.pyapi.object_getattr_string(view_obj, "stop")
    lookup_obj = c.pyapi.object_getattr_string(view_obj, "lookup")

    lookup_val = c.pyapi.to_native_value(LookupType(), lookup_obj).value
    lookup_proxy = c.context.make_helper(c.builder, LookupType(), lookup_val)

    proxyout = c.context.make_helper(c.builder, viewtype)
    proxyout.pos = c.pyapi.number_as_ssize_t(pos_obj)
    proxyout.start = c.pyapi.number_as_ssize_t(start_obj)
    proxyout.stop = c.pyapi.number_as_ssize_t(stop_obj)
    proxyout.arrayptrs = c.context.make_helper(
        c.builder, LookupType.arraytype, lookup_proxy.arrayptrs
    ).data
    proxyout.pylookup = lookup_obj

    c.pyapi.decref(pos_obj)
    c.pyapi.decref(start_obj)
    c.pyapi.decref(stop_obj)
    c.pyapi.decref(lookup_obj)

    if c.context.enable_nrt:
        c.context.nrt.decref(c.builder, LookupType(), lookup_val)

    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)


@numba.extending.box(ArrayViewType)
def box_Array(viewtype, viewval, c):
    print("arrayview_cuda.py line 281: box_Array")
    arrayview_obj = box_ArrayView(viewtype, viewval, c)
    out = c.pyapi.call_method(arrayview_obj, "toarray", ())
    c.pyapi.decref(arrayview_obj)
    return out


def dict2serializable(obj):
    print("arrayview_cuda.py line 289: ...")
    if obj is None:
        return None
    else:
        return tuple(obj.items())


def serializable2dict(obj):
    print("arrayview_cuda.py line 297: ...")
    if obj is None:
        return None
    else:
        return dict(obj)


def box_ArrayView(viewtype, viewval, c):
    print("arrayview_cuda.py line 305: box_ArrayView")
    serializable2dict_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(serializable2dict)
    )
    behavior2_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(dict2serializable(viewtype.behavior))
    )
    behavior_obj = c.pyapi.call_function_objargs(
        serializable2dict_obj, (behavior2_obj,)
    )
    ArrayView_obj = c.pyapi.unserialize(c.pyapi.serialize_object(ArrayView))
    type_obj = c.pyapi.unserialize(c.pyapi.serialize_object(viewtype.type))
    fields_obj = c.pyapi.unserialize(c.pyapi.serialize_object(viewtype.fields))

    proxyin = c.context.make_helper(c.builder, viewtype, viewval)
    pos_obj = c.pyapi.long_from_ssize_t(proxyin.pos)
    start_obj = c.pyapi.long_from_ssize_t(proxyin.start)
    stop_obj = c.pyapi.long_from_ssize_t(proxyin.stop)
    lookup_obj = proxyin.pylookup

    out = c.pyapi.call_function_objargs(
        ArrayView_obj,
        (type_obj, behavior_obj, lookup_obj, pos_obj, start_obj, stop_obj, fields_obj),
    )

    c.pyapi.decref(serializable2dict_obj)
    c.pyapi.decref(behavior2_obj)
    c.pyapi.decref(behavior_obj)
    c.pyapi.decref(ArrayView_obj)
    c.pyapi.decref(type_obj)
    c.pyapi.decref(fields_obj)
    c.pyapi.decref(pos_obj)
    c.pyapi.decref(start_obj)
    c.pyapi.decref(stop_obj)

    return out


@numba.core.typing.templates.infer_global(len)
class type_len(numba.core.typing.templates.AbstractTemplate):
    print("arrayview_cuda.py line 345: class type_len")

    def generic(self, args, kwargs):
        print(
            "arrayview_cuda.py line 347: type_len::generic @numba.core.typing.templates.infer_global(len)"
        )
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], ArrayViewType):
            print("arrayview_cuda.py line 349: args[0]", args[0])
            return numba.intp(args[0])


@numba.extending.lower_builtin(len, ArrayViewType)
def lower_len(context, builder, sig, args):
    print(
        "arrayview_cuda.py line 355: @numba.extending.lower_builtin(len, ArrayViewType)",
        sig,
        sig.args[0],
        args[0],
    )
    proxyin = context.make_helper(builder, sig.args[0], args[0])
    # cgutils.printf(builder, "stop %d, start %d", proxyin.stop, proxyin.start)
    # val = ctypes.cast(x.data.ptr, ctypes.py_object).value
    # printimpl.print_varargs(context, builder, sig, proxyin.stop) # 'LoadInstr' object is not iterable
    return builder.sub(proxyin.stop, proxyin.start)


# @overload_method(ArrayViewType, "len", target="cuda")
# def ArrayViewType_len(array_view):
#    if isinstance(array_view, ArrayViewType):
#        def impl(array_view):
#            return array_view.stop - array_view.start
#        return impl


@numba.core.typing.templates.infer_global(operator.getitem)
class type_getitem(numba.core.typing.templates.AbstractTemplate):
    print("arrayview_cuda.py line 371: type_getitem")

    def generic(self, args, kwargs):
        print(
            "arrayview_cuda.py line 373: @numba.core.typing.templates.infer_global generic class type_getitem"
        )
        # printimpl.print_varargs(context, builder, sig, args)
        if len(args) == 2 and len(kwargs) == 0 and isinstance(args[0], ArrayViewType):
            viewtype, wheretype = args
            print(
                "arrayview_cuda.py line 376: viewtype, wheretype", viewtype, wheretype
            )
            if isinstance(wheretype, numba.types.Integer):
                print(
                    "arrayview_cuda.py line 378: viewtype.type.getitem_at_check->",
                    viewtype.type,
                )
                return viewtype.type.getitem_at_check(viewtype)(viewtype, wheretype)
            elif (
                isinstance(wheretype, numba.types.SliceType) and not wheretype.has_step
            ):
                return viewtype.type.getitem_range(viewtype)(viewtype, wheretype)
            elif isinstance(wheretype, numba.types.StringLiteral):
                return viewtype.type.getitem_field(viewtype, wheretype.literal_value)(
                    viewtype, wheretype
                )
            else:
                raise TypeError(
                    "only an integer, start:stop range, or a *constant* "
                    "field name string may be used as ak.Array "
                    "slices in compiled code"
                )


@numba.extending.lower_builtin(operator.setitem, ArrayViewType, numba.types.Integer)
def lower_setitem_at(context, builder, sig, args):
    rettype, (viewtype, wheretype) = sig.return_type, sig.args
    viewval, whereval = args
    viewproxy = context.make_helper(builder, viewtype, viewval)
    print(
        "arrayview_cuda.py line 400:  ------->",
        viewval,
        whereval,
        viewtype,
        repr(viewtype),
        viewproxy,
    )
    return viewtype.type.lower_setitem_at_check(
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        wheretype,
        whereval,
        True,
        True,
    )


@numba.extending.lower_builtin(operator.getitem, ArrayViewType, numba.types.Integer)
def lower_getitem_at(context, builder, sig, args):
    print("arrayview_cuda.py line 416: @numba.extending.lower_builtin lower_getitem_at")

    rettype, (viewtype, wheretype) = sig.return_type, sig.args
    viewval, whereval = args
    viewproxy = context.make_helper(builder, viewtype, viewval)
    print("arrayview_cuda.py line 421:  -------> viewval        ", viewval)
    print("                                      whereval       ", whereval)
    print("                                      viewtype       ", viewtype)
    print("                                      repr(viewtype) ", repr(viewtype))
    print("                                      viewproxy      ", viewproxy)
    print("-----------------------------------------------------")
    return viewtype.type.lower_getitem_at_check(
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        wheretype,
        whereval,
        True,
        True,
    )


@numba.extending.lower_builtin(operator.getitem, ArrayViewType, numba.types.slice2_type)
def lower_getitem_range(context, builder, sig, args):
    print("arrayview_cuda.py line 438: ...lower_getitem_range...")
    rettype, (viewtype, wheretype) = sig.return_type, sig.args
    viewval, whereval = args
    viewproxy = context.make_helper(builder, viewtype, viewval)
    whereproxy = context.make_helper(builder, wheretype, whereval)
    return viewtype.type.lower_getitem_range(
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        whereproxy.start,
        whereproxy.stop,
        True,
    )


@numba.extending.lower_builtin(
    operator.getitem, ArrayViewType, numba.types.StringLiteral
)
def lower_getitem_field(context, builder, sig, args):
    print("arrayview_cuda.py line 460: ...lower_getitem_field...")
    _, (viewtype, wheretype) = sig.return_type, sig.args
    viewval, whereval = args
    return viewtype.type.lower_getitem_field(
        context, builder, viewtype, viewval, wheretype.literal_value
    )


@numba.core.typing.templates.infer_getattr
class type_getattr(numba.core.typing.templates.AttributeTemplate):
    print("arrayview_cuda.py line 470: class type_getattr")
    key = ArrayViewType

    def generic_resolve(self, viewtype, attr):
        print("arrayview_cuda.py line 474: type_getattr::generic_resolve")
        if attr == "ndim":
            return numba.intp
        else:
            return viewtype.type.getitem_field(viewtype, attr)


@numba.extending.lower_getattr_generic(ArrayViewType)
def lower_getattr_generic(context, builder, viewtype, viewval, attr):
    print("arrayview_cuda.py line 483: ...")
    if attr == "ndim":
        return context.get_constant(numba.intp, viewtype.type.ndim)
    else:
        return viewtype.type.lower_getitem_field(
            context, builder, viewtype, viewval, attr
        )


class IteratorType(numba.types.common.SimpleIteratorType):
    print("arrayview_cuda.py line 493: class IteratorType")

    def __init__(self, viewtype):
        print("arrayview_cuda.py line 495: IteratorType::__init__")
        super().__init__(
            f"ak.Iterator({viewtype.name})",
            viewtype.type.getitem_at_check(viewtype),
        )
        self.viewtype = viewtype


@numba.core.typing.templates.infer
class type_getiter(numba.core.typing.templates.AbstractTemplate):
    print("arrayview_cuda.py line 505: class type_getiter")
    key = "getiter"

    def generic(self, args, kwargs):
        print("arrayview_cuda.py line 509: type_getiter::generic")
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], ArrayViewType):
            return IteratorType(args[0])(args[0])


@numba.core.datamodel.registry.register_default(IteratorType)
class IteratorModel(numba.core.datamodel.models.StructModel):
    print("arrayview_cuda.py line 516: class IteratorModel")

    def __init__(self, dmm, fe_type):
        print("arrayview_cuda.py line 518: IteratorModel::__init__")
        members = [
            ("view", fe_type.viewtype),
            ("length", numba.intp),
            ("at", numba.types.EphemeralPointer(numba.intp)),
        ]
        super().__init__(dmm, fe_type, members)


@numba.extending.lower_builtin("getiter", ArrayViewType)
def lower_getiter(context, builder, sig, args):
    print("arrayview_cuda.py line 529: ...")
    rettype, (viewtype,) = sig.return_type, sig.args
    (viewval,) = args
    viewproxy = context.make_helper(builder, viewtype, viewval)

    proxyout = context.make_helper(builder, rettype)
    proxyout.view = viewval
    proxyout.length = builder.sub(viewproxy.stop, viewproxy.start)
    proxyout.at = numba.core.cgutils.alloca_once_value(
        builder, context.get_constant(numba.intp, 0)
    )

    if context.enable_nrt:
        context.nrt.incref(builder, viewtype, viewval)

    return numba.core.imputils.impl_ret_new_ref(
        context, builder, rettype, proxyout._getvalue()
    )


@numba.extending.lower_builtin("iternext", IteratorType)
@numba.core.imputils.iternext_impl(numba.core.imputils.RefType.BORROWED)
def lower_iternext(context, builder, sig, args, result):
    print("arrayview_cuda.py line 552: ...")
    (itertype,) = sig.args
    (iterval,) = args
    proxyin = context.make_helper(builder, itertype, iterval)
    at = builder.load(proxyin.at)

    is_valid = builder.icmp_signed("<", at, proxyin.length)
    result.set_valid(is_valid)

    with builder.if_then(is_valid, likely=True):
        result.yield_(
            lower_getitem_at(
                context,
                builder,
                itertype.yield_type(itertype.viewtype, numba.intp),
                (proxyin.view, at),
            )
        )
        nextat = numba.core.cgutils.increment_index(builder, at)
        builder.store(nextat, proxyin.at)


########## RecordView


class RecordView:
    @classmethod
    def fromrecord(cls, record):
        behavior = ak._util.behavior_of(record)
        layout = ak.operations.to_layout(
            record,
            allow_record=True,
            allow_other=False,
            numpytype=(np.number, np.bool_, np.datetime64, np.timedelta64),
        )
        assert isinstance(layout, ak.record.Record)
        arraylayout = layout.array
        return RecordView(
            ArrayView(
                tonumbatype(arraylayout.form),
                behavior,
                ak._lookup.Lookup(arraylayout),
                0,
                0,
                len(arraylayout),
                (),
            ),
            layout.at,
        )

    def __init__(self, arrayview, at):
        self.arrayview = arrayview
        self.at = at

    def torecord(self):
        arraylayout = self.arrayview.toarray().layout
        return ak._util.wrap(
            ak.record.Record(arraylayout, self.at), self.arrayview.behavior
        )


@numba.extending.typeof_impl.register(RecordView)
def typeof_RecordView(obj, c):
    return RecordViewType(numba.typeof(obj.arrayview))


class RecordViewType(numba.types.Type):
    def __init__(self, arrayviewtype):
        super().__init__(name=f"ak.RecordViewType({arrayviewtype.name})")
        self.arrayviewtype = arrayviewtype

    @property
    def behavior(self):
        return self.arrayviewtype.behavior

    @property
    def fields(self):
        return self.arrayviewtype.fields

    def typer_field(self, key):
        return self.arrayviewtype.type.getitem_field_record(self, key)

    def lower_field(self, context, builder, val, key):
        return self.arrayviewtype.type.lower_getitem_field_record(
            context, builder, self, val, key
        )


@numba.extending.register_model(RecordViewType)
class RecordViewModel(numba.core.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("arrayview", fe_type.arrayviewtype), ("at", numba.intp)]
        super().__init__(dmm, fe_type, members)


@numba.core.imputils.lower_constant(RecordViewType)
def lower_const_Record(context, builder, recordviewtype, record):
    arrayview_val = lower_const_view(
        context, builder, recordviewtype.arrayviewtype, record._numbaview.arrayview
    )
    proxyout = context.make_helper(builder, recordviewtype)
    proxyout.arrayview = arrayview_val
    proxyout.at = context.get_constant(numba.intp, record._layout.at)
    return proxyout._getvalue()


@numba.extending.unbox(RecordViewType)
def unbox_RecordView(recordviewtype, recordobj, c):
    recordview_obj = c.pyapi.object_getattr_string(recordobj, "_numbaview")
    arrayview_obj = c.pyapi.object_getattr_string(recordview_obj, "arrayview")
    at_obj = c.pyapi.object_getattr_string(recordview_obj, "at")

    arrayview_val = unbox_ArrayView(
        recordviewtype.arrayviewtype, arrayview_obj, c
    ).value

    proxyout = c.context.make_helper(c.builder, recordviewtype)
    proxyout.arrayview = arrayview_val
    proxyout.at = c.pyapi.number_as_ssize_t(at_obj)

    c.pyapi.decref(recordview_obj)
    c.pyapi.decref(at_obj)

    if c.context.enable_nrt:
        c.context.nrt.decref(c.builder, recordviewtype.arrayviewtype, arrayview_val)

    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)


@numba.extending.box(RecordViewType)
def box_RecordView(recordviewtype, viewval, c):
    RecordView_obj = c.pyapi.unserialize(c.pyapi.serialize_object(RecordView))

    proxyin = c.context.make_helper(c.builder, recordviewtype, viewval)
    arrayview_obj = box_ArrayView(recordviewtype.arrayviewtype, proxyin.arrayview, c)
    at_obj = c.pyapi.long_from_ssize_t(proxyin.at)

    recordview_obj = c.pyapi.call_function_objargs(
        RecordView_obj, (arrayview_obj, at_obj)
    )

    out = c.pyapi.call_method(recordview_obj, "torecord", ())

    c.pyapi.decref(RecordView_obj)
    c.pyapi.decref(arrayview_obj)
    c.pyapi.decref(at_obj)
    c.pyapi.decref(recordview_obj)

    return out


@numba.core.typing.templates.infer_global(operator.getitem)
class type_getitem_record(numba.core.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 2 and len(kwargs) == 0 and isinstance(args[0], RecordViewType):
            recordviewtype, wheretype = args

            if isinstance(wheretype, numba.types.StringLiteral):
                return recordviewtype.arrayviewtype.type.getitem_field_record(
                    recordviewtype, wheretype.literal_value
                )(recordviewtype, wheretype)

            else:
                raise TypeError(
                    "only a *constant* field name string may be used as a "
                    "record slice in compiled code"
                )


@numba.extending.lower_builtin(
    operator.getitem, RecordViewType, numba.types.StringLiteral
)
def lower_getitem_field_record(context, builder, sig, args):
    _, (recordviewtype, wheretype) = sig.return_type, sig.args
    recordviewval, whereval = args
    return recordviewtype.arrayviewtype.type.lower_getitem_field_record(
        context, builder, recordviewtype, recordviewval, wheretype.literal_value
    )


@numba.core.typing.templates.infer_getattr
class type_getattr_record(numba.core.typing.templates.AttributeTemplate):
    key = RecordViewType

    def generic_resolve(self, recordviewtype, attr):
        for methodname, typer, lower in ak._util.numba_methods(
            recordviewtype.arrayviewtype.type, recordviewtype.arrayviewtype.behavior
        ):
            if attr == methodname:

                class type_method(numba.core.typing.templates.AbstractTemplate):
                    key = methodname

                    def generic(self, args, kwargs):
                        if len(kwargs) == 0:
                            sig = typer(recordviewtype, args)  # noqa: B023
                            sig = numba.core.typing.templates.Signature(
                                sig.return_type, sig.args, recordviewtype
                            )
                            numba.extending.lower_builtin(
                                methodname,  # noqa: B023
                                recordviewtype,
                                *[
                                    x.literal_type
                                    if isinstance(x, numba.types.Literal)
                                    else x
                                    for x in args
                                ],
                            )(
                                lower  # noqa: B023
                            )
                            return sig

                return numba.types.BoundFunction(type_method, recordviewtype)

        for attrname, typer, _ in ak._util.numba_attrs(
            recordviewtype.arrayviewtype.type, recordviewtype.arrayviewtype.behavior
        ):
            if attr == attrname:
                return typer(recordviewtype)

        return recordviewtype.typer_field(attr)


@numba.extending.lower_getattr_generic(RecordViewType)
def lower_getattr_generic_record(context, builder, recordviewtype, recordviewval, attr):
    for attrname, typer, lower in ak._util.numba_attrs(
        recordviewtype.arrayviewtype.type, recordviewtype.arrayviewtype.behavior
    ):
        if attr == attrname:
            return lower(
                context,
                builder,
                typer(recordviewtype)(recordviewtype),
                (recordviewval,),
            )

    return recordviewtype.lower_field(context, builder, recordviewval, attr)


def register_unary_operator(unaryop):
    @numba.core.typing.templates.infer_global(unaryop)
    class type_binary_operator(numba.core.typing.templates.AbstractTemplate):
        def generic(self, args, kwargs):
            if len(args) == 1 and len(kwargs) == 0:
                behavior = None

                if isinstance(args[0], RecordViewType):
                    left = args[0].arrayviewtype.type
                    behavior = args[0].arrayviewtype.behavior

                    for typer, lower in ak._util.numba_unaryops(
                        unaryop, left, behavior
                    ):
                        numba.extending.lower_builtin(unaryop, *args)(lower)
                        return typer(unaryop, args[0])


for unaryop in (
    abs,
    operator.inv,
    operator.invert,
    operator.neg,
    operator.not_,
    operator.pos,
    operator.truth,
):
    register_unary_operator(unaryop)


def register_binary_operator(binop):
    @numba.core.typing.templates.infer_global(binop)
    class type_binary_operator(numba.core.typing.templates.AbstractTemplate):
        def generic(self, args, kwargs):
            if len(args) == 2 and len(kwargs) == 0:
                left, right, behavior = None, None, None

                if isinstance(args[0], RecordViewType):
                    left = args[0].arrayviewtype.type
                    behavior = args[0].arrayviewtype.behavior

                if isinstance(args[1], RecordViewType):
                    right = args[1].arrayviewtype.type
                    if behavior is None:
                        behavior = args[1].arrayviewtype.behavior

                if left is not None or right is not None:
                    for typer, lower in ak._util.numba_binops(
                        binop, left, right, behavior
                    ):
                        numba.extending.lower_builtin(binop, *args)(lower)
                        return typer(binop, args[0], args[1])


for binop in (
    operator.add,
    operator.and_,
    operator.contains,
    operator.eq,
    operator.floordiv,
    operator.ge,
    operator.gt,
    operator.le,
    operator.lshift,
    operator.lt,
    operator.mod,
    operator.mul,
    operator.ne,
    operator.or_,
    operator.pow,
    operator.rshift,
    operator.sub,
    operator.truediv,
    operator.xor,
    operator.matmul,
):
    register_binary_operator(binop)


########## __contains__


@numba.extending.overload(operator.contains)
def overload_contains(obj, element):
    if isinstance(obj, (ArrayViewType, RecordViewType)) and (
        (element == numba.types.none)
        or (isinstance(element, (numba.types.Number, numba.types.Boolean)))
        or (
            isinstance(element, numba.types.Optional)
            and isinstance(element.type, (numba.types.Number, numba.types.Boolean))
        )
    ):
        statements = []

        def add_statement(indent, name, arraytype, is_array):
            if is_array:
                statements.append("for x in " + name + ":")
                name = "x"
                indent = indent + "    "

            if isinstance(arraytype, ak._connect.numba.layout.RecordArrayType):
                if arraytype.is_tuple:
                    for fi, ft in enumerate(arraytype.contenttypes):
                        add_statement(indent, name + "[" + repr(fi) + "]", ft, False)
                else:
                    for fn, ft in zip(arraytype.fields, arraytype.contenttypes):
                        add_statement(indent, name + "[" + repr(fn) + "]", ft, False)

            elif arraytype.ndim == 1 and not arraytype.is_recordtype:
                if arraytype.is_optiontype:
                    statements.append(
                        indent + "if (element is None and {0} is None) or "
                        "({0} is not None and element == {0}): return True".format(name)
                    )
                else:
                    statements.append(indent + f"if element == {name}: return True")

            else:
                if arraytype.is_optiontype:
                    statements.append(
                        indent + "if (element is None and {0} is None) or "
                        "({0} is not None and element in {0}): return True".format(name)
                    )
                else:
                    statements.append(indent + f"if element in {name}: return True")

        if isinstance(obj, ArrayViewType):
            add_statement("", "obj", obj.type, True)
        else:
            add_statement("", "obj", obj.arrayviewtype.type, False)

        return code_to_function(
            """
def contains_impl(obj, element):
    {}
    return False""".format(
                "\n    ".join(statements)
            ),
            "contains_impl",
        )


########## np.array and np.asarray


def array_supported(dtype):
    return dtype in (
        numba.types.boolean,
        numba.types.int8,
        numba.types.int16,
        numba.types.int32,
        numba.types.int64,
        numba.types.uint8,
        numba.types.uint16,
        numba.types.uint32,
        numba.types.uint64,
        numba.types.float32,
        numba.types.float64,
        numba.types.complex64,
        numba.types.complex128,
    ) or isinstance(dtype, (numba.types.NPDatetime, numba.types.NPTimedelta))


@numba.extending.overload(ak.nplikes.numpy.array)
def overload_np_array(array, dtype=None):
    if isinstance(array, ArrayViewType):
        ndim = array.type.ndim
        inner_dtype = array.type.inner_dtype
        if ndim is not None and array_supported(inner_dtype):
            declare_shape = []
            compute_shape = []
            specify_shape = ["len(array)"]
            ensure_shape = []
            array_name = "array"
            for i in range(ndim - 1):
                declare_shape.append(f"shape{i} = -1")
                compute_shape.append(
                    "{}for x{} in {}:".format("    " * i, i, array_name)
                )
                compute_shape.append("{}    if shape{} == -1:".format("    " * i, i))
                compute_shape.append(
                    "{0}        shape{1} = len(x{1})".format("    " * i, i)
                )
                compute_shape.append(
                    "{0}    elif shape{1} != len(x{1}):".format("    " * i, i)
                )
                compute_shape.append(
                    "{}        raise ValueError('cannot convert to NumPy because "
                    "subarray lengths are not regular')".format("    " * i)
                )
                specify_shape.append(f"shape{i}")
                ensure_shape.append("if shape{0} == -1: shape{0} = 0".format(i))
                array_name = f"x{i}"

            fill_array = []
            index = []
            array_name = "array"
            for i in range(ndim):
                fill_array.append(
                    "{0}for i{1}, x{1} in enumerate({2}):".format(
                        "    " * i, i, array_name
                    )
                )
                index.append(f"i{i}")
                array_name = f"x{i}"

            fill_array.append(
                "{}out[{}] = x{}".format("    " * ndim, "][".join(index), ndim - 1)
            )

            return code_to_function(
                """
def array_impl(array, dtype=None):
    {}
    {}
    {}
    out = numpy.zeros(({}), {})
    {}
    return out
""".format(
                    "\n    ".join(declare_shape),
                    "\n    ".join(compute_shape),
                    "\n    ".join(ensure_shape),
                    ", ".join(specify_shape),
                    f"numpy.{inner_dtype}" if dtype is None else "dtype",
                    "\n    ".join(fill_array),
                ),
                "array_impl",
                {"numpy": ak.nplikes.numpy},
            )


@numba.extending.type_callable(ak.nplikes.numpy.asarray)
def type_asarray(context):
    def typer(arrayview):
        if (
            isinstance(arrayview, ArrayViewType)
            and isinstance(arrayview.type, ak._connect.numba.layout.NumpyArrayType)
            and arrayview.type.ndim == 1
            and array_supported(arrayview.type.inner_dtype)
        ):
            return numba.types.Array(arrayview.type.inner_dtype, 1, "C")

    return typer


@numba.extending.lower_builtin(ak.nplikes.numpy.asarray, ArrayViewType)
def lower_asarray(context, builder, sig, args):
    rettype, (viewtype,) = sig.return_type, sig.args
    (viewval,) = args
    viewproxy = context.make_helper(builder, viewtype, viewval)
    assert isinstance(viewtype.type, ak._connect.numba.layout.NumpyArrayType)

    whichpos = ak._connect.numba.layout.posat(
        context, builder, viewproxy.pos, viewtype.type.ARRAY
    )
    arrayptr = ak._connect.numba.layout.getat(
        context, builder, viewproxy.arrayptrs, whichpos
    )

    bitwidth = ak._connect.numba.layout.type_bitwidth(rettype.dtype)
    itemsize = context.get_constant(numba.intp, bitwidth // 8)

    data = numba.core.cgutils.pointer_add(
        builder,
        arrayptr,
        builder.mul(viewproxy.start, itemsize),
        context.get_value_type(numba.types.CPointer(rettype.dtype)),
    )

    shape = context.make_tuple(
        builder,
        numba.types.UniTuple(numba.types.intp, 1),
        (builder.sub(viewproxy.stop, viewproxy.start),),
    )
    strides = context.make_tuple(
        builder,
        numba.types.UniTuple(numba.types.intp, 1),
        (itemsize,),
    )

    out = numba.np.arrayobj.make_array(rettype)(context, builder)
    numba.np.arrayobj.populate_array(
        out,
        data=data,
        shape=shape,
        strides=strides,
        itemsize=itemsize,
        meminfo=None,
        parent=None,
    )
    return out._getvalue()


########## ArrayView Arguments Handler for CUDA JIT

import ctypes  # noqa: F401
from ctypes import addressof

import numba.cuda as nb_cuda

# from numba import cuda, types  # noqa: F401, E402


class ArrayViewArgHandler:
    def prepare_args(self, ty, val, stream, retr):
        if isinstance(val, int):
            format(val, "x")
        if isinstance(val, ak.Array):

            if isinstance(val.layout.nplike, ak.nplikes.Cupy):

                # Use uint64 for start, stop, pos, the array pointers value and the pylookup value
                tys = types.UniTuple(types.uint64, 5)

                dev = cuda.current_context().device
                print("ArrayViewArgHandler::prepare_args line 108:", dev)

                print("CuPy array???", val.layout.data.data)
                # import numba.cuda as nb_cuda
                array_data_view = nb_cuda.as_cuda_array(val.layout.data)
                print("view --->", array_data_view, type(array_data_view))
                print(
                    "val._numbaview.start",
                    val._numbaview.start,
                    type(val._numbaview.start),
                )
                print(
                    "val._numbaview.stop",
                    val._numbaview.stop,
                    type(val._numbaview.stop),
                )
                print(
                    "val._numbaview.pos", val._numbaview.pos, type(val._numbaview.pos)
                )
                c_intp = ctypes.c_ssize_t

                start = addressof(c_intp(val._numbaview.start))
                stop = addressof(c_intp(val._numbaview.stop))
                pos = addressof(c_intp(val._numbaview.pos))
                arrayptrs = val._numbaview.lookup.arrayptrs
                pylookup = addressof(ctypes.c_void_p(0))

                result_ptr = format(arrayptrs.item(), "x")
                print(
                    "arrayview_cuda.py line 1280: ArrayViewArgHandler::prepare_args: about to return from prepare args and arrayptrs is",
                    result_ptr,
                    arrayptrs.item(),
                )
                print(
                    "arrayview_cuda.py line 1281: ArrayViewArgHandler::prepare_args:",
                    arrayptrs.data.ptr,
                )
                print("start = 0x", format(start, "x"))
                print("stop = 0x", format(stop, "x"))
                print("pos = 0x", format(pos, "x"))
                return tys, (start, stop, pos, arrayptrs.item(), pylookup)
            else:
                raise ak._errors.wrap_error(
                    NotImplementedError(
                        f"{repr(val.layout.nplike)} is not implemented for CUDA. Please transfer the array to CUDA backend to "
                        "continue the operation."
                    )
                )

        else:
            return ty, val


array_view_arg_handler = ArrayViewArgHandler()

###class ArrayViewArgHandler:
#    def prepare_args(self, ty, val, stream, retr):
#        if isinstance(val, ak.Array):
#            # Use uint64 for start, stop, pos, the array pointers value and the pylookup value
#            tys = types.UniTuple(types.uint64, 5)
#
#            # ... Copy data to device if necessary...
#            # use similar to: wrap_arg(val).to_device(retr, stream) where
#            #
#            # def wrap_arg(value, default=InOut):
#            #    return value if isinstance(value, ArgHint) else default(value)

#            dev = cuda.current_context().device
#            print(dev)
#            dev_array = ak.to_backend(val, "cuda") # check if it does copy to device?
#            print(">>>>>>>> COPY TO DEVICE! >>>>>>>", dev_array, repr(dev_array), type(dev_array))
#            ary = np.arange(10)
#            d_ary = cuda.to_device(ary)
#            print(d_ary)
#            print(dev)

#            start, stop, pos, arrayptrs, pylookup = lower_const_array_as_const_view(val)

# Retrieve data back to host
#            def retrieve():
#                # Copy dev_array back to host somehow
#                hary = ak.to_backend(dev_array, "cpu") ########d_ary.copy_to_host()
#                print("retrieve", hary, repr(hary), type(hary))
#                return hary ###ak.to_backend(dev_array, "cpu")

#            # Append retrieve function if necessary
#            retr.append(retrieve)

#            return tys, (start, stop, pos, arrayptrs, pylookup)

# print("High level ak.Array to_numpy data pointer is", ak.to_numpy(val).ctypes.data)
# A pointer to the memory area of the array as a Python integer.
# return types.uint64, val.layout._data.ctypes.data
#
# return types.uint64, ak._connect.numba.arrayview.ArrayView.fromarray(val).lookup.arrayptrs.ctypes.data

#        elif isinstance(val, ak._connect.numba.arrayview.ArrayView):

#            tys = types.UniTuple(types.uint64, 5)

#            start, stop, pos, arrayptrs, pylookup = lower_as_const_view(val)
#            print("Got", start, stop, pos, arrayptrs, pylookup)

#            return tys, (start, stop, pos, arrayptrs, pylookup)
#        else:
#            return ty, val

# arg_handler = ArrayViewArgHandler()
