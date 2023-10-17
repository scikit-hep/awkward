# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import sys
import warnings

from packaging.version import parse as parse_version

import awkward as ak
from awkward._behavior import behavior_of
from awkward._layout import wrap_layout

_has_checked_version = False


def _import_numexpr():
    global _has_checked_version
    try:
        import numexpr
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """install the 'numexpr' package with:

    pip install numexpr --upgrade

or

    conda install numexpr"""
        ) from err
    else:
        if not _has_checked_version:
            if parse_version(numexpr.__version__) < parse_version("2.7.1"):
                warnings.warn(
                    "Awkward Array is only known to work with numexpr 2.7.1 or later"
                    f"(you have version {numexpr.__version__})",
                    RuntimeWarning,
                    stacklevel=1,
                )
            _has_checked_version = True
        return numexpr


def getArguments(names, local_dict=None, global_dict=None):
    call_frame = sys._getframe(2)

    clear_local_dict = False
    if local_dict is None:
        local_dict = call_frame.f_locals
        clear_local_dict = True
    try:
        frame_globals = call_frame.f_globals
        if global_dict is None:
            global_dict = frame_globals

        clear_local_dict = clear_local_dict and frame_globals is not local_dict

        arguments = []
        for name in names:
            try:
                a = local_dict[name]
            except KeyError:
                a = global_dict[name]
            arguments.append(a)  # <--- different from NumExpr
    finally:
        if clear_local_dict:
            local_dict.clear()

    return arguments


def evaluate(
    expression, local_dict=None, global_dict=None, order="K", casting="safe", **kwargs
):
    numexpr = _import_numexpr()

    context = numexpr.necompiler.getContext(kwargs)
    expr_key = (expression, tuple(sorted(context.items())))
    if expr_key not in numexpr.necompiler._names_cache:
        numexpr.necompiler._names_cache[expr_key] = numexpr.necompiler.getExprNames(
            expression, context
        )
    names, ex_uses_vml = numexpr.necompiler._names_cache[expr_key]
    arguments = getArguments(names, local_dict, global_dict)

    arrays = [ak.operations.to_layout(x, allow_unknown=True) for x in arguments]

    def action(inputs, **ignore):
        if all(
            isinstance(x, ak.contents.NumpyArray)
            or not isinstance(x, ak.contents.Content)
            for x in inputs
        ):
            input_primitives = [
                x.data if isinstance(x, ak.contents.NumpyArray) else x for x in inputs
            ]
            return (
                ak.contents.NumpyArray(
                    numexpr.evaluate(
                        expression,
                        dict(zip(names, input_primitives)),
                        {},
                        order=order,
                        casting=casting,
                        **kwargs,
                    )
                ),
            )
        else:
            return None

    behavior = behavior_of(*arrays)
    out = ak._broadcasting.broadcast_and_apply(arrays, action, allow_records=False)
    assert isinstance(out, tuple) and len(out) == 1
    return wrap_layout(out[0], behavior)


evaluate.evaluate = evaluate


def re_evaluate(local_dict=None):
    numexpr = _import_numexpr()

    try:
        compiled_ex = numexpr.necompiler._numexpr_last["ex"]  # noqa: F841
    except KeyError as err:
        raise RuntimeError("not a previous evaluate() execution found") from err
    names = numexpr.necompiler._numexpr_last["argnames"]
    arguments = getArguments(names, local_dict)

    arrays = [ak.operations.to_layout(x, allow_unknown=True) for x in arguments]

    def action(inputs, **ignore):
        if all(
            isinstance(x, ak.contents.NumpyArray)
            or not isinstance(x, ak.contents.Content)
            for x in inputs
        ):
            input_primitives = [
                x.data if isinstance(x, ak.contents.NumpyArray) else x for x in inputs
            ]
            return (
                ak.contents.NumpyArray(
                    numexpr.re_evaluate(dict(zip(names, input_primitives)))
                ),
            )
        else:
            return None

    behavior = behavior_of(*arrays)
    out = ak._broadcasting.broadcast_and_apply(arrays, action, allow_records=False)
    assert isinstance(out, tuple) and len(out) == 1
    return wrap_layout(out[0], behavior)
