# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""NumPy ufunc conversion for flat Awkward views in Numba."""

import numba
import numpy
from numba.core import ir, ir_utils, rewrites

# Register Numba's array-expression rewrite before the conversion rewrite below.
from numba.np.ufunc import array_exprs  # noqa: F401

from awkward._connect.numba.arrayview import ArrayViewType, array_supported
from awkward._connect.numba.layout import NumpyArrayType


def _is_flat(value):
    return (
        isinstance(value, ArrayViewType)
        and isinstance(value.type, NumpyArrayType)
        and value.type.ndim == 1
        and array_supported(value.type.inner_dtype)
    )


def _as_numpy(value):
    return value


@numba.extending.overload(_as_numpy)
def _overload_as_numpy(value):
    if _is_flat(value):

        def impl(value):
            return numpy.asarray(value)

        return impl
    else:

        def impl(value):
            return value

        return impl


def _register_unary(function):
    @numba.extending.overload(function)
    def overload(a, out=None):
        if _is_flat(a) and not _is_flat(out):
            if out is None or isinstance(out, numba.types.NoneType):

                def impl(a, out=None):
                    return function(_as_numpy(a))

            else:

                def impl(a, out=None):
                    return function(_as_numpy(a), _as_numpy(out))

            return impl


def _register_binary(function):
    @numba.extending.overload(function)
    def overload(a, b, out=None):
        if (_is_flat(a) or _is_flat(b)) and not _is_flat(out):
            if out is None or isinstance(out, numba.types.NoneType):

                def impl(a, b, out=None):
                    return function(_as_numpy(a), _as_numpy(b))

            else:

                def impl(a, b, out=None):
                    return function(_as_numpy(a), _as_numpy(b), _as_numpy(out))

            return impl


_seen = set()
for _function in tuple(vars(numpy).values()):
    if isinstance(_function, numpy.ufunc) and _function not in _seen:
        _seen.add(_function)
        if _function.nin == 1:
            _register_unary(_function)
        elif _function.nin == 2:
            _register_binary(_function)


def _replace_variables(expression, replacements):
    if isinstance(expression, ir.Var):
        return replacements.get(expression.name, expression)
    if isinstance(expression, tuple):
        operator, operands = expression
        return operator, [
            _replace_variables(operand, replacements) for operand in operands
        ]
    return expression


@rewrites.register_rewrite("after-inference")
class ConvertUfuncArrayViews(rewrites.Rewrite):
    """Convert view operands before Numba lowers fused array expressions.

    Array expression lowering only extracts scalar dtypes from native Arrays;
    passing an ArrayView directly makes it compile the scalar kernel with an
    array argument. Materialize the existing zero-copy np.asarray view in IR.
    """

    def __init__(self, state=None):
        self.typingctx = state.typingctx

    def match(self, func_ir, block, typemap, calltypes):
        self.block = block
        self.typemap = typemap
        self.calltypes = calltypes
        self.matches = {}
        for instruction in block.find_insts(ir.Assign):
            expression = instruction.value
            if isinstance(expression, ir.Expr) and expression.op == "arrayexpr":
                variables = {
                    var.name: var
                    for var in expression.list_vars()
                    if _is_flat(typemap[var.name])
                }
                if variables:
                    self.matches[instruction.target.name] = variables
        return bool(self.matches)

    def apply(self):
        block = self.block.copy()
        block.body = []
        for instruction in self.block.body:
            if (
                isinstance(instruction, ir.Assign)
                and instruction.target.name in self.matches
            ):
                replacements = {}
                for name, variable in self.matches[instruction.target.name].items():
                    loc, scope = instruction.loc, instruction.target.scope
                    function = ir.Var(scope, ir_utils.mk_unique_var("ak_asarray"), loc)
                    converted = ir.Var(
                        scope, ir_utils.mk_unique_var("ak_numpy_view"), loc
                    )
                    self.typemap[function.name] = self.typingctx.resolve_value_type(
                        numpy.asarray
                    )
                    call = ir.Expr.call(function, [variable], (), loc)
                    signature = self.typingctx.resolve_function_type(
                        numpy.asarray, (self.typemap[name],), {}
                    )
                    self.typemap[converted.name] = signature.return_type
                    self.calltypes[call] = signature
                    block.body.extend(
                        [
                            ir.Assign(
                                ir.Global("asarray", numpy.asarray, loc), function, loc
                            ),
                            ir.Assign(call, converted, loc),
                        ]
                    )
                    replacements[name] = converted

                instruction.value.expr = _replace_variables(
                    instruction.value.expr, replacements
                )
            block.body.append(instruction)
        return block
