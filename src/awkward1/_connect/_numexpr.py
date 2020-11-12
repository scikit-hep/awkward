# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import warnings
import sys
import distutils.version

import awkward1.layout
import awkward1.operations.convert
import awkward1._util


checked_version = False


def import_numexpr():
    global checked_version
    try:
        import numexpr
    except ImportError:
        raise ImportError(
            """install the 'numexpr' package with:

    pip install numexpr --upgrade

or

    conda install numexpr"""
        )
    else:
        if not checked_version and distutils.version.LooseVersion(
            numexpr.__version__
        ) < distutils.version.LooseVersion("2.7.1"):
            warnings.warn(
                "awkward1 is only known to work with numexpr 2.7.1 or later"
                "(you have version {0})".format(numexpr.__version__),
                RuntimeWarning,
            )
        checked_version = True
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
    numexpr = import_numexpr()

    context = numexpr.necompiler.getContext(kwargs, frame_depth=1)
    expr_key = (expression, tuple(sorted(context.items())))
    if expr_key not in numexpr.necompiler._names_cache:
        numexpr.necompiler._names_cache[expr_key] = numexpr.necompiler.getExprNames(
            expression, context
        )
    names, ex_uses_vml = numexpr.necompiler._names_cache[expr_key]
    arguments = getArguments(names, local_dict, global_dict)

    arrays = [
        awkward1.operations.convert.to_layout(x, allow_record=True, allow_other=True)
        for x in arguments
    ]

    def getfunction(inputs, depth):
        if all(
            isinstance(x, awkward1.layout.NumpyArray)
            or not isinstance(x, awkward1.layout.Content)
            for x in inputs
        ):
            return lambda: (
                awkward1.layout.NumpyArray(
                    numexpr.evaluate(
                        expression,
                        dict(zip(names, inputs)),
                        {},
                        order=order,
                        casting=casting,
                        **kwargs
                    )
                ),
            )
        else:
            return None

    behavior = awkward1._util.behaviorof(*arrays)
    out = awkward1._util.broadcast_and_apply(
        arrays, getfunction, behavior, allow_records=False
    )
    assert isinstance(out, tuple) and len(out) == 1
    return awkward1._util.wrap(out[0], behavior)


evaluate.evaluate = evaluate


def re_evaluate(local_dict=None):
    numexpr = import_numexpr()

    try:
        compiled_ex = numexpr.necompiler._numexpr_last["ex"]  # noqa: F841
    except KeyError:
        raise RuntimeError(
            "not a previous evaluate() execution found"
            + awkward1._util.exception_suffix(__file__)
        )
    names = numexpr.necompiler._numexpr_last["argnames"]
    arguments = getArguments(names, local_dict)

    arrays = [
        awkward1.operations.convert.to_layout(x, allow_record=True, allow_other=True)
        for x in arguments
    ]

    def getfunction(inputs, depth):
        if all(
            isinstance(x, awkward1.layout.NumpyArray)
            or not isinstance(x, awkward1.layout.Content)
            for x in inputs
        ):
            return lambda: (
                awkward1.layout.NumpyArray(
                    numexpr.re_evaluate(dict(zip(names, inputs)))
                ),
            )
        else:
            return None

    behavior = awkward1._util.behaviorof(*arrays)
    out = awkward1._util.broadcast_and_apply(
        arrays, getfunction, behavior, allow_records=False
    )
    assert isinstance(out, tuple) and len(out) == 1
    return awkward1._util.wrap(out[0], behavior)


evaluate.re_evaluate = re_evaluate
