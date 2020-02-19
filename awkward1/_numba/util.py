# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import numpy
import numba

py27 = (sys.version_info[0] < 3)

if not py27:
    exec("""
def debug(context, builder, *args):
    assert len(args) % 2 == 0
    tpes, vals = args[0::2], args[1::2]
    context.get_function(print, numba.none(*tpes))(builder, tuple(vals))
""", globals())

RefType = numba.int64

index8tpe = numba.types.Array(numba.int8, 1, "C")
indexU8tpe = numba.types.Array(numba.uint8, 1, "C")
index32tpe = numba.types.Array(numba.int32, 1, "C")
indexU32tpe = numba.types.Array(numba.uint32, 1, "C")
index64tpe = numba.types.Array(numba.int64, 1, "C")
def indextpe(indexname):
    if indexname == "64":
        return index64tpe
    elif indexname == "32":
        return index32tpe
    elif indexname == "U32":
        return indexU32tpe
    elif indexname == "8":
        return index8tpe
    elif indexname == "U8":
        return indexU8tpe
    else:
        raise AssertionError("unrecognized index type: {0}".format(indexname))

def cast(context, builder, fromtpe, totpe, val):
    if isinstance(fromtpe, llvmlite.ir.types.IntType):
        if fromtpe.width == 8:
            fromtpe = numba.int8
        elif fromtpe.width == 16:
            fromtpe = numba.int16
        elif fromtpe.width == 32:
            fromtpe = numba.int32
        elif fromtpe.width == 64:
            fromtpe = numba.int64
        else:
            raise AssertionError("unrecognized bitwidth")
    if fromtpe.bitwidth < totpe.bitwidth:
        return builder.sext(val, context.get_value_type(totpe))
    elif fromtpe.bitwidth > totpe.bitwidth:
        return builder.trunc(val, context.get_value_type(totpe))
    else:
        return val

def arraylen(context, builder, tpe, val, totpe=None):
    if isinstance(tpe, numba.types.Array):
        out = numba.targets.arrayobj.array_len(context, builder, numba.intp(tpe), (val,))
    else:
        out = tpe.lower_len(context, builder, numba.intp(tpe), (val,))
    if totpe is None:
        return out
    else:
        return cast(context, builder, numba.intp, totpe, out)

def dict2items(obj):
    if obj is None:
        return obj
    else:
        return tuple(sorted(obj.items()))

def items2str(items):
    return "{" + ", ".join("{0}: {1}".format(n, x) for n, x in items) + "}"

def items2dict_impl(c, items):
    if items is None:
        return c.pyapi.make_none()
    else:
        dict_obj = c.pyapi.unserialize(c.pyapi.serialize_object(dict))
        items_obj = c.pyapi.unserialize(c.pyapi.serialize_object(items))
        out = c.pyapi.call_function_objargs(dict_obj, (items_obj,))
        c.pyapi.decref(dict_obj)
        c.pyapi.decref(items_obj)
        return out
