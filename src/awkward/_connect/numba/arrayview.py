# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import operator

import numba
import numba.core.typing
import numba.core.typing.ctypes_utils

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def code_to_function(code, function_name, externals=None, debug=False):
    if debug:
        print("################### " + function_name)  # noqa: T201
        print(code)  # noqa: T201
    namespace = {} if externals is None else dict(externals)
    exec(code, namespace)
    return namespace[function_name]


def tonumbatype(form):
    if isinstance(form, ak.forms.EmptyForm):
        return tonumbatype(form.to_NumpyForm(np.dtype(np.float64)))

    elif isinstance(form, ak.forms.NumpyForm):
        if len(form.inner_shape) == 0:
            return ak._connect.numba.layout.NumpyArrayType.from_form(form)
        else:
            return tonumbatype(form.to_RegularForm())

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
    return LookupType()


class LookupType(numba.types.Type):
    arraytype = numba.types.Array(numba.intp, 1, "C")

    def __init__(self):
        super().__init__(name="ak.LookupType()")


@numba.extending.register_model(LookupType)
class LookupModel(numba.core.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("arrayptrs", fe_type.arraytype)]
        super().__init__(dmm, fe_type, members)


@numba.extending.unbox(LookupType)
def unbox_Lookup(lookuptype, lookupobj, c):
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
        behavior = ak._util.behavior_of(array)
        layout = ak.operations.to_layout(
            array,
            allow_record=False,
            allow_other=False,
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
        self.type = type
        self.behavior = behavior
        self.lookup = lookup
        self.pos = pos
        self.start = start
        self.stop = stop
        self.fields = fields

    def toarray(self):
        layout = self.type.tolayout(self.lookup, self.pos, self.fields)
        sliced = layout._getitem_range(slice(self.start, self.stop))
        return ak._util.wrap(sliced, self.behavior)


@numba.extending.typeof_impl.register(ArrayView)
def typeof_ArrayView(obj, c):
    return ArrayViewType(obj.type, obj.behavior, obj.fields)


def wrap(type, viewtype, fields):
    if fields is None:
        return ArrayViewType(type, viewtype.behavior, viewtype.fields)
    else:
        return ArrayViewType(type, viewtype.behavior, fields)


def repr_behavior(behavior):
    return repr(behavior)


class ArrayViewType(numba.types.IterableType, numba.types.Sized):
    def __init__(self, type, behavior, fields):
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
        return IteratorType(self)


@numba.extending.register_model(ArrayViewType)
class ArrayViewModel(numba.core.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
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
    return lower_const_view(context, builder, viewtype, array._numbaview)


def lower_const_view(context, builder, viewtype, view):
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
    view_obj = c.pyapi.object_getattr_string(arrayobj, "_numbaview")
    out = unbox_ArrayView(viewtype, view_obj, c)
    c.pyapi.decref(view_obj)
    return out


def unbox_ArrayView(viewtype, view_obj, c):
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
    arrayview_obj = box_ArrayView(viewtype, viewval, c)
    out = c.pyapi.call_method(arrayview_obj, "toarray", ())
    c.pyapi.decref(arrayview_obj)
    return out


def dict2serializable(obj):
    if obj is None:
        return None
    else:
        return tuple(obj.items())


def serializable2dict(obj):
    if obj is None:
        return None
    else:
        return dict(obj)


def box_ArrayView(viewtype, viewval, c):
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
    def generic(self, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], ArrayViewType):
            return numba.intp(args[0])


@numba.extending.lower_builtin(len, ArrayViewType)
def lower_len(context, builder, sig, args):
    proxyin = context.make_helper(builder, sig.args[0], args[0])
    return builder.sub(proxyin.stop, proxyin.start)


@numba.core.typing.templates.infer_global(operator.getitem)
class type_getitem(numba.core.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 2 and len(kwargs) == 0 and isinstance(args[0], ArrayViewType):
            viewtype, wheretype = args
            if isinstance(wheretype, numba.types.Integer):
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


@numba.extending.lower_builtin(operator.getitem, ArrayViewType, numba.types.Integer)
def lower_getitem_at(context, builder, sig, args):
    rettype, (viewtype, wheretype) = sig.return_type, sig.args
    viewval, whereval = args
    viewproxy = context.make_helper(builder, viewtype, viewval)
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
    _, (viewtype, wheretype) = sig.return_type, sig.args
    viewval, whereval = args
    return viewtype.type.lower_getitem_field(
        context, builder, viewtype, viewval, wheretype.literal_value
    )


@numba.core.typing.templates.infer_getattr
class type_getattr(numba.core.typing.templates.AttributeTemplate):
    key = ArrayViewType

    def generic_resolve(self, viewtype, attr):
        if attr == "ndim":
            return numba.intp
        else:
            return viewtype.type.getitem_field(viewtype, attr)


@numba.extending.lower_getattr_generic(ArrayViewType)
def lower_getattr_generic(context, builder, viewtype, viewval, attr):
    if attr == "ndim":
        return context.get_constant(numba.intp, viewtype.type.ndim)
    else:
        return viewtype.type.lower_getitem_field(
            context, builder, viewtype, viewval, attr
        )


class IteratorType(numba.types.common.SimpleIteratorType):
    def __init__(self, viewtype):
        super().__init__(
            f"ak.Iterator({viewtype.name})",
            viewtype.type.getitem_at_check(viewtype),
        )
        self.viewtype = viewtype


@numba.core.typing.templates.infer
class type_getiter(numba.core.typing.templates.AbstractTemplate):
    key = "getiter"

    def generic(self, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], ArrayViewType):
            return IteratorType(args[0])(args[0])


@numba.core.datamodel.registry.register_default(IteratorType)
class IteratorModel(numba.core.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("view", fe_type.viewtype),
            ("length", numba.intp),
            ("at", numba.types.EphemeralPointer(numba.intp)),
        ]
        super().__init__(dmm, fe_type, members)


@numba.extending.lower_builtin("getiter", ArrayViewType)
def lower_getiter(context, builder, sig, args):
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
        layout = ak.operations.to_layout(record, allow_record=True, allow_other=False)
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


@numba.extending.overload(ak._nplikes.numpy.array)
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
                {"numpy": ak._nplikes.numpy},
            )


@numba.extending.type_callable(ak._nplikes.numpy.asarray)
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


@numba.extending.lower_builtin(ak._nplikes.numpy.asarray, ArrayViewType)
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
