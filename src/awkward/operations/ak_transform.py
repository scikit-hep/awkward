# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import copy
from functools import reduce

import awkward as ak
from awkward._backends.numpy import NumpyBackend
from awkward._broadcasting import (
    apply_step as apply_broadcasting_step,
)
from awkward._broadcasting import (
    broadcast_pack,
    broadcast_unpack,
)
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, ensure_same_backend
from awkward._namedaxis import NAMED_AXIS_KEY, NamedAxesWithDims, _unify_named_axis

__all__ = ("transform",)

cpu = NumpyBackend.instance()


@high_level_function()
def transform(
    transformation,
    array,
    *more_arrays,
    depth_context=None,
    lateral_context=None,
    allow_records=True,
    broadcast_parameters_rule="intersect",
    left_broadcast=True,
    right_broadcast=True,
    numpy_to_regular=False,
    regular_to_jagged=False,
    return_value="simplified",
    expect_return_value=False,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """
    Args:
        transformation (callable): Function to apply to each node of the array.
            See below for details.
        array: Array-like data (anything #ak.to_layout recognizes), but not an
            #ak.Record or #ak.record.Record.
        more_arrays: Additional arrays to be broadcasted together (with first `array`)
            and used together in the transformation. See below for details.
        depth_context (None or dict): User data to propagate through the transformation.
            New data added to `depth_context` is available to the entire *subtree*
            at which it is added, but no other *subtrees*. For example, data added
            during the transformation will not be in the original `depth_context`
            after the transformation.
        lateral_context (None or dict): User data to propagate through the transformation.
            New data added to `lateral_context` is available at any later step of
            the depth-first walk over the tree, including *other subtrees*. For
            example, data added during the transformation will be in the original
            `lateral_context` after the transformation.
        allow_records (bool): If False and the recursive walk encounters any
            #ak.contents.RecordArray nodes, an error is raised.
        broadcast_parameters_rule (str): Rule for broadcasting parameters, one of:
            - `"intersect"`
            - `"all_or_nothing"`
            - `"one_to_one"`
            - `"none"`
        left_broadcast (bool): If `more_arrays` are provided, the parameter
            determines whether the arrays are left-broadcasted, which is
            Awkward-like broadcasting.
        right_broadcast (bool): If `more_arrays` are provided, the parameter
            determines whether the arrays are right-broadcasted, which is
            NumPy-like broadcasting.
        numpy_to_regular (bool): If True, multidimensional #ak.contents.NumpyArray
            nodes are converted into #ak.contents.RegularArray nodes before
            calling `transformation`.
        regular_to_jagged (bool): If True, regular-type lists are converted into
            variable-length lists before calling `transformation`.
        return_value (`"none"`, `"original", `"simplified"`): If `"none"`, the output of
            this function is None; if `"original"`, untouched nodes surrounding
            the ones replaced by the `transformation` are returned in their original
            state; if `"simplified"`, the #ak.Content.simplified constructor is
            used on the surrounding nodes to ensure that option-type and union-type
            nodes are not nested inappropriately. Note that if `return_value` is `"none"`,
            the only way to get information out of this function is through the
            `lateral_context`.
        expect_return_value (bool): If True, raise a `RuntimeError` if the transformer
            does not terminate the recursion.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Applies a `transformation` function to every node of an Awkward array or arrays
    to either obtain a transformed copy or extract data from a walk over the arrays'
    low-level layout nodes.

    This is a public interface to the infrastructure that is used to implement most
    Awkward Array operations. As such, it's very powerful, but low-level.

    Here is a "hello world" example:

        >>> def say_hello(layout, depth, **kwargs):
        ...     print("Hello", type(layout).__name__, "at", depth)
        ...
        >>> array = ak.Array([[1.1, 2.2, "three"], [], None, [4.4, 5.5]])
        >>> ak.transform(say_hello, array, return_value="none")
        Hello IndexedOptionArray at 1
        Hello ListOffsetArray at 1
        Hello UnionArray at 2
        Hello NumpyArray at 2
        Hello ListOffsetArray at 2
        Hello NumpyArray at 3

    In the above, `say_hello` is called on every node of the `array`, which has
    a lot of nodes because it has nested lists, missing data, and a union of
    different types. The data types are low-level "layouts," subclasses of
    #ak.contents.Content, rather than high-level #ak.Array.

    The primary purpose of this function is to allow you to edit one level of
    structure without having to worry about what it's embedded in. Suppose, for
    instance, you want to apply NumPy's `np.round` function to numerical data,
    regardless of what lists or other structures they're embedded in.

    The return value must be a subclass of #ak.contents.Content (to replace the
    array node) or None (to leave the array node unchanged).

        >>> def rounder(layout, **kwargs):
        ...     if layout.is_numpy:
        ...         return ak.contents.NumpyArray(
        ...             np.round(layout.data).astype(np.int32)
        ...         )
        ...
        >>> array = ak.Array(
        ... [[[[[1.1, 2.2, 3.3], []], None], []],
        ...  [[[[4.4, 5.5]]]]]
        ... )
        >>> ak.transform(rounder, array).show(type=True)
        type: 2 * var * var * option[var * var * int32]
        [[[[[1, 2, 3], []], None], []],
         [[[[4, 6]]]]]

    If you pass multiple arrays to this function (`more_arrays`), those arrays
    will be broadcasted and all inputs, at the same level of depth and structure,
    will be passed to the `transformation` function as a group.

    Here is an example with broadcasting:

        >>> def combine(layouts, **kwargs):
        ...     assert len(layouts) == 2
        ...     if layouts[0].is_numpy and layouts[1].is_numpy:
        ...         return ak.contents.NumpyArray(
        ...             layouts[0].data + 10 * layouts[1].data
        ...         )
        ...
        >>> array1 = ak.Array([[1, 2, 3], [], None, [4, 5]])
        >>> array2 = ak.Array([1, 2, 3, 4])
        >>> ak.transform(combine, array1, array2)
        <Array [[11, 12, 13], [], None, [44, 45]] type='4 * option[var * int64]'>

    The `1` and `4` from `array2` are broadcasted to the `[1, 2, 3]` and the
    `[4, 5]` of `array1`, and the other elements disappear because they are
    broadcasted with an empty list and a missing value. Note that the first argument
    of this `transformation` function is a *list* of layouts, not a single layout.
    There are always 2 layouts because 2 arrays were passed to #ak.transform.

    Signature of the transformation function
    ========================================

    If there is only one array, the first argument of `transformation` is a
    #ak.contents.Content instance. If there are multiple arrays (`more_arrays`),
    the first argument is a list of #ak.contents.Content instances.

    All other arguments can be absorbed into a `**kwargs` because they will always
    be passed to your function by keyword. They are

    * depth (int): The current list depth, where 1 is the outermost array and
        higher numbers are deeper levels of list nesting. This does not count
        nesting of other data structures, such as option-types and records.
    * depth_context (None or dict): Any user-specified data. You can add to
        this dict during transformation; changes would only be seen in the
        subtree's nodes.
    * lateral_context (None or dict): Any user-specified data. You can add to
        this dict during transformation; changes would be seen in any node
        visited later in the depth-first search.
    * continuation (callable): Zero-argument function that continues the
        recursion from this point in the walk, so that you can perform
        post-processing instead of pre-processing.

    For completeness, the following arguments are also passed to `transformation`,
    but you usually won't need them:

    * behavior (None or dict): Behavior that would be attached to the output
        array(s) if `highlevel`.
    * backend (array library / kernel library shim): Handle to the NumPy
        library, CuPy, etc., depending on the type of arrays.
    * options (dict): Options provided to #ak.transform.

    If there is only one array, the `transformation` function must either return
    None or return an #ak.contents.Content.

    If there are multiple arrays (`more_arrays`), then the transformation function
    may return one array or a tuple of arrays. (The preferred type is a tuple, even
    if it has length 1.)

    The final return value of #ak.transform is a new array or tuple of arrays
    constructed by replacing nodes when `transformation` returns a
    #ak.contents.Content or tuple of #ak.contents.Content, and leaving
    nodes unchanged when `transformation` returns None. If `transformation` returns
    length-1 tuples, the final output is an array, not a length-1 tuple.

    If `return_value` is `"none"`, #ak.transform returns None. This is useful for
    functions that return non-array data through `lateral_context`. The other two
    choices, `"original"` and `"simplified"`, determine how untouched array nodes,
    the ones that are _not_ modified by the `transformation` function, are returned.
    With `"original"`, they are returned without modification, which might result
    in illegal combinations of option-type and union-type, which would raise an
    error. With `"simplified"`, the surrounding array nodes are simplified upon
    reconstruction. For example, if the `transformation` puts a new #ak.contents.ByteMaskedArray
    inside an existing #ak.contents.ByteMaskedArray, the two will be consolidated
    into a single option-type array node.

    Contexts
    ========

    The `depth_context` and `lateral_context` allow you to pass your own data into
    the transformation as well as communicate between calls of `transformation` on
    different nodes. The `depth_context` limits this communication to descendants
    of the subtree in which the data were added; `lateral_context` does not have
    this limit. (`depth_context` is shallow-copied at each node during descent;
    `lateral_context` is never copied.)

    For example, consider this array:

        >>> array = ak.Array([
        ...     [{"x": [1], "y": 1.1}, {"x": [1, 2], "y": 2.2}, {"x": [1, 2, 3], "y": 3.3}],
        ...     [],
        ...     [{"x": [1, 2, 3, 4], "y": 4.4}, {"x": [1, 2, 3, 4, 5], "y": 5.5}],
        ... ])

    If we accumulate node type names using `depth_context`,

        >>> def crawl(layout, depth_context, **kwargs):
        ...     depth_context["types"] = depth_context["types"] + (type(layout).__name__,)
        ...     print(depth_context["types"])
        ...
        >>> context = {"types": ()}
        >>> ak.transform(crawl, array, depth_context=context, return_value="none")
        ('ListOffsetArray',)
        ('ListOffsetArray', 'RecordArray')
        ('ListOffsetArray', 'RecordArray', 'ListOffsetArray')
        ('ListOffsetArray', 'RecordArray', 'ListOffsetArray', 'NumpyArray')
        ('ListOffsetArray', 'RecordArray', 'NumpyArray')
        >>> context
        {'types': ()}

    The data in `depth_context["types"]` represents a path from the root of the
    tree to the current node. There is never, for instance, more than one leaf-type
    (#ak.contents.NumpyArray) in the tuple. Also, the `context` is unchanged
    outside of the function.

    On the other hand, if we do the same with a `lateral_context`,

        >>> def crawl(layout, lateral_context, **kwargs):
        ...     lateral_context["types"] = lateral_context["types"] + (type(layout).__name__,)
        ...     print(lateral_context["types"])
        ...
        >>> context = {"types": ()}
        >>> ak.transform(crawl, array, lateral_context=context, return_value="none")
        ('ListOffsetArray',)
        ('ListOffsetArray', 'RecordArray')
        ('ListOffsetArray', 'RecordArray', 'ListOffsetArray')
        ('ListOffsetArray', 'RecordArray', 'ListOffsetArray', 'NumpyArray')
        ('ListOffsetArray', 'RecordArray', 'ListOffsetArray', 'NumpyArray', 'NumpyArray')
        >>> context
        {'types': ('ListOffsetArray', 'RecordArray', 'ListOffsetArray', 'NumpyArray', 'NumpyArray')}

    The data accumulate through the walk over the tree. There are two leaf-types
    (#ak.contents.NumpyArray) in the tuple because this tree has two leaves.
    The data are even available outside of the function, so `lateral_context` can
    be paired with `return_value="none"` to extract non-array data, rather than
    transforming the array.

    The visitation order is stable: a recursive walk always proceeds through the
    same tree in the same order.

    Continuation
    ============

    The `transformation` function is given an input, untransformed layout or layouts.
    Some algorithms need to perform a correction on transformed outputs, so
    `continuation()` can be called at any point to continue descending but obtain
    the transformed result.

    For example, this function inserts an option-type at every level of an array:

        >>> def insert_optiontype(layout, continuation, **kwargs):
        ...     return ak.contents.UnmaskedArray(continuation())
        ...
        >>> array = ak.Array([[[[[1.1, 2.2, 3.3], []]], []], [[[[4.4, 5.5]]]]])
        >>> array.type.show()
        2 * var * var * var * var * float64

        >>> array2 = ak.transform(insert_optiontype, array)
        >>> array2.type.show()
        2 * option[var * option[var * option[var * option[var * ?float64]]]]

    In the original array, every node is a #ak.contents.ListOffsetArray except
    the leaf, which is a #ak.contents.NumpyArray. The call to `continuation()`
    returns a #ak.contents.ListOffsetArray with its contents transformed, which
    is the argument of a new #ak.contents.UnmaskedArray.

    To see this process as it happens, we can add `print` statements to the function.

        >>> def insert_optiontype(input, continuation, **kwargs):
        ...     print("before", input.form.type)
        ...     output = ak.contents.UnmaskedArray(continuation())
        ...     print("after ", output.form.type)
        ...     return output
        ...
        >>> ak.transform(insert_optiontype, array)
        before var * var * var * var * float64
        before var * var * var * float64
        before var * var * float64
        before var * float64
        before float64
        after  ?float64
        after  option[var * ?float64]
        after  option[var * option[var * ?float64]]
        after  option[var * option[var * option[var * ?float64]]]
        after  option[var * option[var * option[var * option[var * ?float64]]]]
        <Array [[[[[1.1, ..., 3.3], ...]], ...], ...] type='2 * option[var * option...'>

    Broadcasting
    ============

    When multiple arrays are provided (`more_arrays`), all of the arrays are
    broadcasted during the walk so that the `transformation` function is eventually
    provided with a list of layouts that have compatible types (for mathematical
    operations, etc.).

    For instance, given these two arrays:

        >>> array1 = ak.Array([[1, 2, 3], [], None, [4, 5]])
        >>> array2 = ak.Array([10, 20, 30, 40])

    The following single-array function shows the nodes encountered when walking
    down either one of them.

        >>> def one_array(layout, **kwargs):
        ...     print(type(layout).__name__)
        ...
        >>> ak.transform(one_array, array1, return_value="none")
        IndexedOptionArray
        ListOffsetArray
        NumpyArray
        >>> ak.transform(one_array, array2, return_value="none")
        NumpyArray

    The first array has three nested nodes; the second has only one node.

    However, when the following two-array function is applied,

        >>> def two_arrays(layouts, **kwargs):
        ...     assert len(layouts) == 2
        ...     print(type(layouts[0]).__name__, ak.to_list(layouts[0]))
        ...     print(type(layouts[1]).__name__, ak.to_list(layouts[1]))
        ...     print()
        ...
        >>> ak.transform(two_arrays, array1, array2)
        RegularArray [[[1, 2, 3], [], None, [4, 5]]]
        RegularArray [[10, 20, 30, 40]]

        IndexedOptionArray [[1, 2, 3], [], None, [4, 5]]
        NumpyArray [10, 20, 30, 40]

        ListArray [[1, 2, 3], [], [4, 5]]
        NumpyArray [10, 20, 40]

        NumpyArray [1, 2, 3, 4, 5]
        NumpyArray [10, 10, 10, 40, 40]

        (<Array [[1, 2, 3], [], None, [4, 5]] type='4 * option[var * int64]'>,
         <Array [[10, 10, 10], [], None, [40, 40]] type='4 * option[var * int64]'>)

    The incompatible types of the two arrays eventually becomes the same type by
    duplicating and removing values wherever necessary. If you cannot perform an
    operation on a #ak.contents.ListArray and a #ak.contents.NumpyArray,
    wait for a later iteration, in which both will be #ak.contents.NumpyArray
    (if the original arrays are broadcastable).

    The return value, without transformation, is the same as what
    #ak.broadcast_arrays would return. See #ak.broadcast_arrays for an
    explanation of `left_broadcast` and `right_broadcast`.

    Broadcasting Parameters
    =======================

    When broadcasting multiple arrays with parameters, there are different ways of
    assigning parameters to the outputs. The assignment of array parameters happens
    at every level above the transformation action.

    The method of parameter assignment used by the broadcasting routine is controlled
    by the `broadcast_parameters_rule` option, which can take one of the following
    values:

    `"intersect"`
        The parameters of each output array will correspond to the intersection
        of the parameters from each of the input arrays.

    `"all_or_nothing"`
        If the parameters of the input arrays are all equal, then they will be used
        for each output array. Otherwise, the output arrays will not be given
        parameters.

    `"one_to_one"`
        If the number of output arrays matches the number of input arrays, then the
        output arrays are given the parameters of the input arrays. Otherwise, a
        ValueError is raised.

    `"none"`
        The output arrays will not be given parameters.


    Performance Tip
    ================

    #ak.transform will traverse the layout of (potentially multiple) arrays once.
    This can be useful if one wants to apply a batch of transformations in one single
    layout traversal. Traversing the layout multiple times can be inefficient.

    Consider the following example:

        >>> def batch_of_operations(array):
        ...     return np.sqrt(np.sin(array) + 1) - 1
        ...
        >>> def apply_batch_of_operations(layout, **kwargs):
        ...     if layout.is_numpy:
        ...         return ak.contents.NumpyArray(
        ...             batch_of_operations(layout.data)
        ...         )
        ...
        >>> array = ak.Array(
        ... [[[[[1.1, 2.2, 3.3], []], None], []],
        ...  [[[[4.4, 5.5]]]]]
        ... )
        >>> %timeit ak.transform(apply_batch_of_operations, array)
        ... 68.5 μs ± 663 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
        >>> %timeit batch_of_operations(array)
        ... 1.07 ms ± 39.1 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

    The first `%timeit` cell shows the time it takes to apply the batch of operations using #ak.transform,
    which allows to apply the operations in one single traversal of the layout. The second `%timeit` cell shows
    the runtime of applying the operations directly to the array, which traverses the layout multiple times.
    To be more explicit: one layout traversal for each operation.


    See also: #ak.is_valid and #ak.valid_when to check the validity of transformed
    outputs.
    """
    # Dispatch
    yield (array, *more_arrays)

    # Implementation
    return _impl(
        transformation,
        array,
        more_arrays,
        depth_context,
        lateral_context,
        allow_records,
        broadcast_parameters_rule,
        left_broadcast,
        right_broadcast,
        numpy_to_regular,
        regular_to_jagged,
        return_value,
        expect_return_value,
        behavior,
        highlevel,
        attrs,
    )


def _impl(
    transformation,
    array,
    more_arrays,
    depth_context,
    lateral_context,
    allow_records,
    broadcast_parameters_rule,
    left_broadcast,
    right_broadcast,
    numpy_to_regular,
    regular_to_jagged,
    return_value,
    expect_return_value,
    behavior,
    highlevel,
    attrs,
):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layouts = ensure_same_backend(
            ctx.unwrap(
                array,
                allow_record=False,
                allow_unknown=False,
                none_policy="error",
                regulararray=True,
                use_from_iter=True,
                primitive_policy="error",
                string_policy="as-characters",
            ),
            *(
                ctx.unwrap(
                    x,
                    allow_record=False,
                    allow_unknown=False,
                    none_policy="error",
                    regulararray=True,
                    use_from_iter=True,
                    primitive_policy="error",
                    string_policy="as-characters",
                )
                for x in more_arrays
            ),
        )

    options = {
        "allow_records": allow_records,
        "left_broadcast": left_broadcast,
        "right_broadcast": right_broadcast,
        "numpy_to_regular": numpy_to_regular,
        "regular_to_jagged": regular_to_jagged,
        "keep_parameters": True,
        "return_simplified": return_value == "simplified",
        "return_array": return_value != "none",
        "function_name": "ak.transform",
        "broadcast_parameters_rule": broadcast_parameters_rule,
        "expect_return_value": expect_return_value,
    }

    transformer_did_terminate = False

    if len(layouts) == 1:

        def action(layout, **kwargs):
            nonlocal transformer_did_terminate
            out = transformation(layout, **kwargs, behavior=behavior)

            if out is None:
                return out

            elif isinstance(out, ak.contents.Content):
                transformer_did_terminate = True
                return out

            else:
                raise TypeError(
                    f"transformation must return a Content or None, not {type(out)}\n\n{out!r}"
                )

        # An exception to the rule of ak._do.recursively_apply, for symmetry with
        # ak._broadcasting.apply_step, below. ak_transform._impl knows implementation details.
        out = layouts[0]._recursively_apply(
            action,
            1,
            copy.copy(depth_context),
            lateral_context,
            options,
        )

        if return_value == "none":
            return None
        elif expect_return_value and not transformer_did_terminate:
            raise RuntimeError(
                "the transformation function was expected to terminate by returning a Content, "
                "but instead only returned None."
            )
        else:
            return ctx.wrap(out, highlevel=highlevel)

    else:

        def action(inputs, **kwargs):
            nonlocal transformer_did_terminate
            out = transformation(tuple(inputs), **kwargs, behavior=behavior)

            if out is None:
                if all(isinstance(x, ak.contents.NumpyArray) for x in inputs):
                    return tuple(inputs)
                else:
                    return None

            elif isinstance(out, tuple):
                transformer_did_terminate = True
                for x in out:
                    if not isinstance(x, ak.contents.Content):
                        raise TypeError(
                            f"transformation must return a Content, tuple of Contents, or None, "
                            f"not a tuple containing {type(x)}\n\n{x!r}"
                        )
                return out

            elif isinstance(out, ak.contents.Content):
                transformer_did_terminate = True
                return (out,)

            else:
                raise TypeError(
                    f"transformation must return a Content, tuple of Contents, or None, not {type(out)}\n\n{out!r}"
                )

        if depth_context is None:
            depth_context = {}
        if lateral_context is None:
            lateral_context = {}
        assert NAMED_AXIS_KEY not in depth_context
        assert NAMED_AXIS_KEY not in lateral_context
        _depth_context, _lateral_context = NamedAxesWithDims.prepare_contexts(
            [array, *more_arrays]
        )
        depth_context.update(_depth_context)
        lateral_context.update(_lateral_context)
        backend = next((layout.backend for layout in layouts), cpu)
        isscalar = []
        out = apply_broadcasting_step(
            backend,
            broadcast_pack(layouts, isscalar),
            action,
            0,
            copy.copy(depth_context),
            lateral_context,
            options,
        )
        assert isinstance(out, tuple)
        out = [broadcast_unpack(x, isscalar) for x in out]

        # Unify named axes propagated through the broadcast
        out_named_axis = reduce(
            _unify_named_axis, lateral_context[NAMED_AXIS_KEY].named_axis
        )

        if return_value == "none":
            return
        elif expect_return_value and not transformer_did_terminate:
            raise RuntimeError(
                "the transformation function was expected to terminate by returning a Content, "
                "or tuple of Contents, but instead only returned None."
            )
        elif len(out) == 1:
            wrapped_out = ctx.wrap(out[0], highlevel=highlevel)
            return ak.operations.ak_with_named_axis._impl(
                wrapped_out,
                named_axis=out_named_axis,
                highlevel=highlevel,
                behavior=ctx.behavior,
                attrs=ctx.attrs,
            )
        else:
            wrapped_out = []
            for x in out:
                wrapped = ctx.wrap(x, highlevel=highlevel)
                wrapped_out.append(
                    ak.operations.ak_with_named_axis._impl(
                        wrapped,
                        named_axis=out_named_axis,
                        highlevel=highlevel,
                        behavior=ctx.behavior,
                        attrs=ctx.attrs,
                    )
                )
            return tuple(wrapped_out)
