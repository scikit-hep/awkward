# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import copy
from collections.abc import Iterable

import awkward as ak


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
    return_array=True,
    highlevel=True,
    behavior=None,
):
    """
    Args:
        transformation (callable): Function to apply to each node of the array.
            See below for details.
        array: First (and possibly only) array to be transformed. Can be any
            array-like object that #ak.to_layout recognizes, but not an
            #ak.Record.
        more_arrays: Arrays to be broadcasted together (with first `array`) and
            used together in the transformation. See below for details.
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
            #ak._v2.contents.RecordArray nodes, an error is raised.
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
        numpy_to_regular (bool): If True, multidimensional #ak._v2.contents.NumpyArray
            nodes are converted into #ak._v2.contents.RegularArray nodes before
            calling `transformation`.
        regular_to_jagged (bool): If True, regular-type lists are converted into
            variable-length lists before calling `transformation`.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
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
        >>> array = ak._v2.Array([[1.1, 2.2, "three"], [], None, [4.4, 5.5]])
        >>>
        >>> ak._v2.transform(say_hello, array, return_array=False)
        Hello IndexedOptionArray at 1
        Hello ListOffsetArray at 1
        Hello UnionArray at 2
        Hello NumpyArray at 2
        Hello ListOffsetArray at 2
        Hello NumpyArray at 3

    In the above, `say_hello` is called on every node of the `array`, which has
    a lot of nodes because it has nested lists, missing data, and a union of
    different types. The data types are low-level "layouts," subclasses of
    #ak._v2.contents.Content, rather than high-level #ak.Array.

    The primary purpose of this function is to allow you to edit one level of
    structure without having to worry about what it's embedded in. Suppose, for
    instance, you want to apply NumPy's `np.round` function to numerical data,
    regardless of what lists or other structures they're embedded in.

        >>> def rounder(layout, **kwargs):
        ...     if layout.is_NumpyType:
        ...         return np.round(layout.data).astype(np.int32)
        ...
        >>> array = ak._v2.Array(
        ... [[[[[1.1, 2.2, 3.3], []], None], []],
        ...  [[[[4.4, 5.5]]]]]
        ... )
        >>>
        >>> ak._v2.transform(rounder, array).show(type=True)
        type: 2 * var * var * option[var * var * int32]
        [[[[[1, 2, 3], []], None], []],
         [[[[4, 6]]]]]

    If you pass multiple arrays to this function (`more_arrays`), those arrays
    will be broadcasted and all inputs, at the same level of depth and structure,
    will be passed to the `transformation` function as a group.

    Here is an example with broadcasting:

        >>> def combine(layouts, **kwargs):
        ...     assert len(layouts) == 2
        ...     if layouts[0].is_NumpyType and layouts[1].is_NumpyType:
        ...         return layouts[0].data + 10 * layouts[1].data
        ...
        >>> array1 = ak._v2.Array([[1, 2, 3], [], None, [4, 5]])
        >>> array2 = ak._v2.Array([1, 2, 3, 4])
        >>>
        >>> ak._v2.transform(combine, array1, array2)
        <Array [[11, 12, 13], [], None, [44, 45]] type='4 * option[var * int64]'>

    The `1` and `4` from `array2` are broadcasted to the `[1, 2, 3]` and the
    `[4, 5]` of `array1`, and the other elements disappear because they are
    broadcasted with an empty list and a missing value. Note that the first argument
    of this `transformation` function is a *list* of layouts, not a single layout.
    There are always 2 layouts because 2 arrays were passed to #ak._v2.transform.

    Signature of the transformation function
    ========================================

    If there is only one array, the first argument of `transformation` is a
    #ak._v2.contents.Content instance. If there are multiple arrays (`more_arrays`),
    the first argument is a list of #ak._v2.contents.Content instances.

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
       * nplike (array library shim): Handle to the NumPy library, CuPy, etc.,
           depending on the type of arrays.
       * options (dict): Options provided to #ak._v2.transform.

    If there is only one array, the `transformation` function must either return
    None or return an array: #ak._v2.contents.Content, #ak._v2.Array, or a NumPy,
    CuPy, etc. array. (The preferred type is #ak._v2.contents.Content; all others
    are converted to a layout.)

    If there are multiple arrays (`more_arrays`), then the transformation function
    may return one array or a tuple of arrays. (The preferred type is a tuple, even
    if it has length 1.)

    The final return value of #ak._v2.transform is a new array or tuple of arrays
    constructed by replacing nodes when `transformation` returns a
    #ak._v2.contents.Content or tuple of #ak._v2.contents.Content, and leaving
    nodes unchanged when `transformation` returns None. If `transformation` returns
    length-1 tuples, the final output is an array, not a length-1 tuple.

    If `return_array` is False, #ak._v2.transform returns None. This is useful for
    functions that return non-array data through `lateral_context`.

    Contexts
    ========

    The `depth_context` and `lateral_context` allow you to pass your own data into
    the transformation as well as communicate between calls of `transformation` on
    different nodes. The `depth_context` limits this communication to descendants
    of the subtree in which the data were added; `lateral_context` does not have
    this limit. (`depth_context` is shallow-copied at each node during descent;
    `lateral_context` is never copied.)

    For example, consider this array:

        >>> array = ak._v2.Array([
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
        >>> ak._v2.transform(crawl, array, depth_context=context, return_array=False)
        ('ListOffsetArray',)
        ('ListOffsetArray', 'RecordArray')
        ('ListOffsetArray', 'RecordArray', 'ListOffsetArray')
        ('ListOffsetArray', 'RecordArray', 'ListOffsetArray', 'NumpyArray')
        ('ListOffsetArray', 'RecordArray', 'NumpyArray')
        >>> context
        {'types': ()}

    The data in `depth_context["types"]` represents a path from the root of the
    tree to the current node. There is never, for instance, more than one leaf-type
    (#ak._v2.contents.NumpyArray) in the tuple. Also, the `context` is unchanged
    outside of the function.

    On the other hand, if we do the same with a `lateral_context`,

        >>> context = {"types": ()}
        >>> ak._v2.transform(crawl, array, lateral_context=context, return_array=False)
        ('ListOffsetArray',)
        ('ListOffsetArray', 'RecordArray')
        ('ListOffsetArray', 'RecordArray', 'ListOffsetArray')
        ('ListOffsetArray', 'RecordArray', 'ListOffsetArray', 'NumpyArray')
        ('ListOffsetArray', 'RecordArray', 'ListOffsetArray', 'NumpyArray', 'NumpyArray')
        >>> context
        {'types': ('ListOffsetArray', 'RecordArray', 'ListOffsetArray', 'NumpyArray', 'NumpyArray')}

    The data accumulate through the walk over the tree. There are two leaf-types
    (#ak._v2.contents.NumpyArray) in the tuple because this tree has two leaves.
    The data are even available outside of the function, so `lateral_context` can
    be paired with `return_array=False` to extract non-array data, rather than
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
        ...     return ak._v2.contents.UnmaskedArray(continuation())
        ...
        >>> array = ak._v2.Array([[[[[1.1, 2.2, 3.3], []]], []], [[[[4.4, 5.5]]]]])
        >>> array.type.show()
        2 * var * var * var * var * float64
        >>>
        >>> array2 = ak._v2.transform(insert_optiontype, array)
        >>> array2.type.show()
        2 * option[var * option[var * option[var * option[var * ?float64]]]]

    In the original array, every node is a #ak._v2.contents.ListOffsetArray except
    the leaf, which is a #ak._v2.contents.NumpyArray. The call to `continuation()`
    returns a #ak._v2.contents.ListOffsetArray with its contents transformed, which
    is the argument of a new #ak._v2.contents.UnmaskedArray.

    To see this process as it happens, we can add `print` statements to the function.

        >>> def insert_optiontype(input, continuation, **kwargs):
        ...     print("before", input.form.type)
        ...     output = ak._v2.contents.UnmaskedArray(continuation())
        ...     print("after ", output.form.type)
        ...     return output
        ...
        >>> ak._v2.transform(insert_optiontype, array)
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

        >>> array1 = ak._v2.Array([[1, 2, 3], [], None, [4, 5]])
        >>> array2 = ak._v2.Array([10, 20, 30, 40])

    The following single-array function shows the nodes encountered when walking
    down either one of them.

        >>> def one_array(layout, **kwargs):
        ...     print(type(layout).__name__)
        ...
        >>> ak._v2.transform(one_array, array1, return_array=False)
        IndexedOptionArray
        ListOffsetArray
        NumpyArray
        >>> ak._v2.transform(one_array, array2, return_array=False)
        NumpyArray

    The first array has three nested nodes; the second has only one node.

    However, when the following two-array function is applied,

        >>> def two_arrays(layouts, **kwargs):
        ...     assert len(layouts) == 2
        ...     print(type(layouts[0]).__name__, ak._v2.to_list(layouts[0]))
        ...     print(type(layouts[1]).__name__, ak._v2.to_list(layouts[1]))
        ...     print()
        ...
        >>> ak._v2.transform(two_arrays, array1, array2)
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
    operation on a #ak._v2.contents.ListArray and a #ak._v2.contents.NumpyArray,
    wait for a later iteration, in which both will be #ak._v2.contents.NumpyArray
    (if the original arrays are broadcastable).

    The return value, without transformation, is the same as what
    #ak._v2.broadcast_arrays would return. See #ak._v2.broadcast_arrays for an
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

    See also: #ak.is_valid and #ak.valid_when to check the validity of transformed
    outputs.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.transform",
        dict(
            transformation=transformation,
            array=array,
            more_arrays=more_arrays,
            depth_context=depth_context,
            lateral_context=lateral_context,
            allow_records=allow_records,
            broadcast_parameters_rule=broadcast_parameters_rule,
            left_broadcast=left_broadcast,
            right_broadcast=right_broadcast,
            numpy_to_regular=numpy_to_regular,
            regular_to_jagged=regular_to_jagged,
            return_array=return_array,
            behavior=behavior,
            highlevel=highlevel,
        ),
    ):
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
            return_array,
            behavior,
            highlevel,
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
    return_array,
    behavior,
    highlevel,
):
    behavior = ak._v2._util.behavior_of(*((array,) + more_arrays), behavior=behavior)

    layout = ak._v2.to_layout(array, allow_record=False, allow_other=False)
    more_layouts = [
        ak._v2.to_layout(x, allow_record=False, allow_other=False) for x in more_arrays
    ]
    nplike = ak.nplike.of(layout, *more_layouts)

    options = {
        "allow_records": allow_records,
        "left_broadcast": left_broadcast,
        "right_broadcast": right_broadcast,
        "numpy_to_regular": numpy_to_regular,
        "regular_to_jagged": regular_to_jagged,
        "keep_parameters": True,
        "return_array": return_array,
        "function_name": "ak._v2.transform",
        "broadcast_parameters_rule": broadcast_parameters_rule,
    }

    if len(more_layouts) == 0:

        def action(layout, **kwargs):
            out = transformation(layout, **kwargs)

            if out is None:
                return out

            if isinstance(out, ak._v2.highlevel.Array):
                return out.layout

            if isinstance(out, ak._v2.contents.Content):
                return out

            if hasattr(out, "dtype") and hasattr(out, "shape"):
                return ak._v2.contents.NumpyArray(out)

            else:
                raise ak._v2._util.error(
                    TypeError(
                        f"transformation must return an Awkward array, not {type(out)}\n\n{out!r}"
                    )
                )

        out = layout._recursively_apply(
            action,
            behavior,
            1,
            copy.copy(depth_context),
            lateral_context,
            options,
        )

        if return_array:
            return ak._v2._util.wrap(out, behavior, highlevel)

    else:

        def action(inputs, **kwargs):
            out = transformation(inputs, **kwargs)

            if out is None:
                if all(isinstance(x, ak._v2.contents.NumpyArray) for x in inputs):
                    return tuple(inputs)
                else:
                    return None

            if isinstance(out, ak._v2.highlevel.Array):
                return (out.layout,)

            if isinstance(out, ak._v2.contents.Content):
                return (out,)

            if hasattr(out, "dtype") and hasattr(out, "shape"):
                return (ak._v2.contents.NumpyArray(out),)

            if isinstance(out, Iterable) and not isinstance(out, tuple):
                out = tuple(out)

            if any(isinstance(x, ak._v2.highlevel.Array) for x in out):
                out = tuple(
                    x.layout if isinstance(x, ak._v2.highlevel.Array) else x
                    for x in out
                )

            if any(hasattr(x, "dtype") and hasattr(x, "shape") for x in out):
                out = tuple(
                    x.layout if hasattr(x, "dtype") and hasattr(x, "shape") else x
                    for x in out
                )

            for x in out:
                if not isinstance(x, ak._v2.contents.Content):
                    raise ak._v2._util.error(
                        TypeError(
                            f"transformation must return an Awkward array or tuple of arrays, not {type(x)}\n\n{x!r}"
                        )
                    )

            return out

        inputs = [layout] + more_layouts
        isscalar = []
        out = ak._v2._broadcasting.apply_step(
            nplike,
            ak._v2._broadcasting.broadcast_pack(inputs, isscalar),
            action,
            0,
            copy.copy(depth_context),
            lateral_context,
            behavior,
            options,
        )
        assert isinstance(out, tuple)
        out = [ak._v2._broadcasting.broadcast_unpack(x, isscalar, nplike) for x in out]

        if return_array:
            if len(out) == 1:
                return ak._v2._util.wrap(out[0], behavior, highlevel)
            else:
                return tuple(ak._v2._util.wrap(x, behavior, highlevel) for x in out)
