# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def argcombinations(
    array,
    n,
    *,
    replacement=False,
    axis=1,
    fields=None,
    parameters=None,
    with_name=None,
    highlevel=True,
    behavior=None,
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        n (int): The number of items to choose from each list: `2` chooses
            unique pairs, `3` chooses unique triples, etc.
        replacement (bool): If True, combinations that include the same
            item more than once are allowed; otherwise each item in a
            combinations is strictly unique.
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        fields (None or list of str): If None, the pairs/triples/etc. are
            tuples with unnamed fields; otherwise, these `fields` name the
            fields. The number of `fields` must be equal to `n`.
        parameters (None or dict): Parameters for the new
            #ak.contents.RecordArray node that is created by this operation.
        with_name (None or str): Assigns a `"__record__"` name to the new
            #ak.contents.RecordArray node that is created by this operation
            (overriding `parameters`, if necessary).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Computes a Cartesian product (i.e. cross product) of `array` with itself
    that is restricted to combinations sampled without replacement,
    like #ak.combinations, but returning integer indexes for
    #ak.Array.__getitem__.

    The motivation and uses of this function are similar to those of
    #ak.argcartesian. See #ak.combinations and #ak.argcartesian for a more
    complete description.
    """
    with ak._errors.OperationErrorContext(
        "ak.argcombinations",
        dict(
            array=array,
            n=n,
            replacement=replacement,
            axis=axis,
            fields=fields,
            parameters=parameters,
            with_name=with_name,
            highlevel=highlevel,
            behavior=behavior,
        ),
    ):
        return _impl(
            array,
            n,
            replacement,
            axis,
            fields,
            parameters,
            with_name,
            highlevel,
            behavior,
        )


def _impl(
    array, n, replacement, axis, fields, parameters, with_name, highlevel, behavior
):
    if parameters is None:
        parameters = {}
    else:
        parameters = dict(parameters)
    if with_name is not None:
        parameters["__record__"] = with_name

    if axis < 0:
        raise ak._errors.wrap_error(
            ValueError("the 'axis' for argcombinations must be non-negative")
        )
    else:
        layout = ak._do.local_index(
            ak.operations.to_layout(array, allow_record=False, allow_other=False),
            axis,
        )
        out = ak._do.combinations(
            layout,
            n,
            replacement=replacement,
            axis=axis,
            fields=fields,
            parameters=parameters,
        )
        return ak._util.wrap(out, behavior, highlevel, like=array)
