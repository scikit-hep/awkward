# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""NumPy ufunc conversion for flat Awkward views in Numba."""

import numba
import numpy
from numba.core import ir, ir_utils, rewrites
from numba.np import ufunc_db

# Register Numba's array-expression rewrite before the conversion rewrite below.
from numba.np.ufunc import array_exprs  # noqa: F401

from awkward._connect.numba.arrayview import is_flat_numpy_view


def _as_numpy(value):
    return value


@numba.extending.overload(_as_numpy)
def _overload_as_numpy(value):
    if is_flat_numpy_view(value):

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
        if is_flat_numpy_view(a) and not is_flat_numpy_view(out):
            if out is None:

                def impl(a, out=None):
                    return function(_as_numpy(a))

            elif isinstance(out, numba.types.Array):

                def impl(a, out=None):
                    return function(_as_numpy(a), _as_numpy(out))

            else:
                return None

            return impl


def _register_binary(function):
    @numba.extending.overload(function)
    def overload(a, b, out=None):
        if (is_flat_numpy_view(a) or is_flat_numpy_view(b)) and not is_flat_numpy_view(
            out
        ):
            if out is None:

                def impl(a, b, out=None):
                    return function(_as_numpy(a), _as_numpy(b))

            elif isinstance(out, numba.types.Array):

                def impl(a, b, out=None):
                    return function(_as_numpy(a), _as_numpy(b), _as_numpy(out))

            else:
                return None

            return impl


def _register_binary_two_outputs(function):
    @numba.extending.overload(function)
    def overload(a, b, *outputs):
        if not (is_flat_numpy_view(a) or is_flat_numpy_view(b)):
            return None
        if len(outputs) == 0:

            def impl(a, b, *outputs):
                return function(_as_numpy(a), _as_numpy(b))

            return impl
        elif len(outputs) == 2 and all(
            isinstance(output, numba.types.Array) for output in outputs
        ):

            def impl(a, b, *outputs):
                return function(_as_numpy(a), _as_numpy(b), outputs[0], outputs[1])

            return impl
        return None


def _register_ufuncs():
    # Register only native implementations, not unsupported NumPy ufuncs or
    # generalized ufuncs. Aliases share the same ufunc object.
    for function in ufunc_db.get_ufuncs():
        if function.signature is not None:
            continue
        if function.nout == 1:
            if function.nin == 1:
                _register_unary(function)
            elif function.nin == 2:
                _register_binary(function)
        elif function.nin == 2 and function.nout == 2:
            _register_binary_two_outputs(function)


_register_ufuncs()


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
        self.has_flat_views = any(
            is_flat_numpy_view(value) for value in state.typemap.values()
        )

    def match(self, func_ir, block, typemap, calltypes):
        if not self.has_flat_views:
            return False
        self.func_ir = func_ir
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
                    if is_flat_numpy_view(typemap[var.name])
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
                    global_value = ir.Global("asarray", numpy.asarray, loc)
                    block.body.extend(
                        [
                            ir.Assign(global_value, function, loc),
                            ir.Assign(call, converted, loc),
                        ]
                    )
                    self.func_ir._definitions[function.name] = [global_value]
                    self.func_ir._definitions[converted.name] = [call]
                    replacements[name] = converted

                # Keep the input block intact and register the replacement's
                # definition for subsequent inlining and parallel passes.
                keywords = dict(instruction.value._kws)
                keywords["expr"] = _replace_variables(
                    instruction.value.expr, replacements
                )
                expression = ir.Expr(
                    instruction.value.op, instruction.value.loc, **keywords
                )
                instruction = ir.Assign(expression, instruction.target, instruction.loc)
                self.func_ir._definitions[instruction.target.name] = [expression]
            block.body.append(instruction)
        return block
