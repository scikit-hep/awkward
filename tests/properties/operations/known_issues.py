# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Predicates for known issues; the property tests skip affected draws.

Each issue has one public predicate, named `has_issue_<number>` after
the issue it detects, with the description and link in its docstring;
an issue not yet reported takes a placeholder number (`NNN1`, `NNN2`,
...). A predicate takes only the array — nothing else decides whether
an array is affected — so any operation's test module can use it. A
test module calls the predicates it needs from its private
`_would_raise_from_known_issue` function and applies any option
conditions there (a negative `axis` for `ak.flatten`, for example). A
predicate may over-match — an extra skipped draw is harmless, and a
missed one fails the tests — so widen a trigger rather than narrow
it. Delete a predicate in the pull request that fixes its issue: the
skipped draws are then tested again. If an issue is resolved as
intended behavior instead, the exclusion moves into the design
predicate (`_should_not_raise`) of each affected operation.
"""

import awkward as ak


def has_issue_4260(a: ak.Array) -> bool:
    """Return `True` if a record spanning more than one depth is in the array.

    Such a record makes the form branch, so a negative `axis` does
    not resolve; `ak.flatten` then skips its documented
    `AxisError`/`ValueError` paths, and
    `RecordArray._offsets_and_flattened` raises a bare
    `AssertionError` ("RecordArray content with axis > depth + 1
    returned a non-empty offsets from offsets_and_flattened"). The
    depths are those of each record's whole subtree, so a union of
    unequal depths below a record counts. Reported:
    https://github.com/scikit-hep/awkward/issues/4260
    """
    stack = [a.layout.form]
    while stack:
        node = stack.pop()
        if node.is_record:
            lo, hi = node.minmax_depth
            if lo != hi:
                return True
            stack.extend(node.contents)
        elif node.is_union:
            stack.extend(node.contents)
        elif not (node.is_numpy or node.is_unknown):
            stack.append(node.content)
    return False


def has_issue_4259(a: ak.Array) -> bool:
    """Return `True` if a `float16` or `float128` leaf is in the array.

    `ak.sort`, `ak.argsort`, and every reducer except `ak.count`
    raise `KeyError` from the kernel-table lookup when a `float16`
    leaf reaches a kernel (`ak.sum`, `ak.min`, and `ak.max` reach one
    only in a partial reduction; a whole-array reduction takes a
    NumPy path): awkward-cpp has no `float16` kernels, and the dtype
    is neither mapped onto an existing kernel
    (`KernelReducer._dtype_for_kernel` does this for datetimes and
    complex numbers) nor refused with a documented error. `float128`
    fails the same lookup (as `numpy.longdouble` in the kernel key)
    on the platforms where NumPy provides it — continuous
    integration on Linux found it; macOS NumPy has no `float128`.
    Which leaves reach a kernel depends on the operation and the
    layout, so every matching leaf counts here, although one that
    never reaches a kernel works. Reported:
    https://github.com/scikit-hep/awkward/issues/4259
    """
    stack = [a.layout.form]
    while stack:
        node = stack.pop()
        if node.is_numpy:
            if node.primitive in ("float16", "float128"):
                return True
        elif node.is_record or node.is_union:
            stack.extend(node.contents)
        elif not node.is_unknown:
            stack.append(node.content)
    return False


def has_issue_4261(a: ak.Array) -> bool:
    """Return `True` if flattening all levels would merge non-promotable leaves.

    `ak.flatten(axis=None)` and `ak.ravel` flatten every branch to
    its leaves and merge them with `ak._do.mergemany`. A record or
    union in the form can combine leaves from different families:
    integers (with bool, without `uint64`), `uint64` on its own,
    floats (with complex), linear-unit timedeltas, calendar-unit
    timedeltas (months, years), datetimes (all units), and strings
    (with bytestrings). Integers promote with floats and with either
    timedelta family; `uint64` promotes with the integers and the
    floats but with neither timedelta family; any other
    combination of two or more families makes the merge fail with a
    bare `AssertionError` ("cannot merge NumpyArray with
    ListOffsetArray"), an `AttributeError`, NumPy's
    `DTypePromotionError`, or a `TypeError` from datetime-unit
    metadata, depending on the layout, instead of building a union or
    raising a documented error.

    The families are modeled on NumPy's documented promotion rules,
    not on the current implementation, except that datetimes with
    timedeltas, which NumPy promotes, are grouped as non-promotable;
    `np.result_type` is still not a substitute (it fails on extreme
    unit pairs that merge). Unprobed combinations may still fail the
    tests; such a failure indicates this trigger is missing a
    combination.
    Reported: https://github.com/scikit-hep/awkward/issues/4261
    """
    form = a.layout.form
    if not _contains_record_or_union(form):
        return False
    families = _leaf_families(form)
    promotable_pairs = (
        {"integer", "uint64", "float"},
        {"integer", "timedelta"},
        {"integer", "timedelta-calendar"},
    )
    return len(families) > 1 and not any(families <= pair for pair in promotable_pairs)


def has_issue_4262(a: ak.Array) -> bool:
    """Return `True` if a regular list of variable-length lists is under a union.

    To restore the interleaving order of a union,
    `UnionArray._remove_structure` broadcasts each branch against its
    elements' positions with `ak.broadcast_arrays`
    (`ak.flatten(axis=None)` and `ak.ravel` reach it), and the
    broadcast raises `ValueError` ("cannot broadcast ListArray of
    length 3 with NumpyArray of length 1") on a regular list of
    variable-length lists. The failure is in the broadcasting itself,
    with no union among its inputs (related to issue #4247); only a
    union causes the flattening to broadcast. The trigger is such a
    regular list anywhere under a union node, descending through
    list, option, indexed, and record nodes; a string does not count
    as a variable-length list. Which configurations reach the
    broadcast depends on the current implementation's internals, so
    every such union matches here, although a regular list of size 1,
    or one that no values reach, happens to work. Reported:
    https://github.com/scikit-hep/awkward/issues/4262
    """
    stack = [(a.layout.form, False)]
    while stack:
        node, in_union = stack.pop()
        if node.parameter("__array__") in ("string", "bytestring"):
            continue
        if node.is_union:
            stack.extend((c, True) for c in node.contents)
        elif node.is_record:
            stack.extend((c, in_union) for c in node.contents)
        elif node.is_numpy or node.is_unknown:
            continue
        else:
            if in_union and node.is_regular and _is_variable_length_list(node.content):
                return True
            stack.append((node.content, in_union))
    return False


def has_issue_4263(a: ak.Array) -> bool:
    """Return `True` if a union that no values reach is in the array.

    `ak.flatten(axis=None)` and `ak.ravel` collect the values of
    every branch and merge them with `ak._do.mergemany`; a union that
    no values reach — judged on reachable values, so an option mask
    or an empty list above the union counts — contributes no parts,
    and the merge fails its `assert len(contents) != 0` instead of
    returning an empty result. Which branch types the collection
    keeps depends on the current implementation's internals, so every
    such union matches here, although one with only plain numeric
    branches happens to work. Reported:
    https://github.com/scikit-hep/awkward/issues/4263
    """
    return _reaches_empty_union(a.layout)


def has_issue_4264(a: ak.Array) -> bool:
    """Return `True` if an option type sits under an untrimmed regular list.

    The caller applies the option condition (for `ak.all`, an integer
    `axis`). A `RegularArray` may validly carry content longer than
    `size * length`, and the reducers mishandle an option node inside
    such untrimmed content: an option directly above a list fails a
    bare `AssertionError` in `RegularArray._reduce_next` on every
    run, and an option above a leaf raises `IndexError` from an
    out-of-range carry index nondeterministically (`ak.all`,
    `ak.any`, `ak.sum`, `ak.count`, and `ak.min` are all affected).
    With content trimmed to `size * length` the same reductions
    succeed. Reported:
    https://github.com/scikit-hep/awkward/issues/4264
    """
    stack = [(a.layout, False)]
    while stack:
        node, in_untrimmed = stack.pop()
        if node.is_numpy or node.is_unknown:
            continue
        if node.parameter("__array__") in ("string", "bytestring"):
            continue
        if node.is_option and in_untrimmed:
            return True
        if node.is_regular and node.content.length > node.size * node.length:
            in_untrimmed = True
        if node.is_record or node.is_union:
            stack.extend((c, in_untrimmed) for c in node.contents)
        else:
            stack.append((node.content, in_untrimmed))
    return False


def _is_variable_length_list(form: ak.forms.Form) -> bool:
    """Return `True` if the node has `var` type.

    Descends through option and indexed nodes; a string is not a
    list here (the `__array__` parameter overrides the structure).
    """
    node = form
    while node.is_option or node.is_indexed:
        node = node.content
    if node.parameter("__array__") in ("string", "bytestring"):
        return False
    return node.is_list and not node.is_regular


def _contains_record_or_union(form: ak.forms.Form) -> bool:
    """Return `True` if a record or union is somewhere in the tree.

    A string node is a leaf, not structure: the descent stops there.
    """
    stack = [form]
    while stack:
        node = stack.pop()
        if node.parameter("__array__") in ("string", "bytestring"):
            continue
        if node.is_record or node.is_union:
            return True
        if not (node.is_numpy or node.is_unknown):
            stack.append(node.content)
    return False


def _leaf_families(form: ak.forms.Form) -> set[str]:
    """Return the merge families of the leaf types.

    The family names and their groupings are described in
    `has_issue_4261`. A string counts as one leaf, not as a list of
    characters.
    """
    families = set()
    stack = [form]
    while stack:
        node = stack.pop()
        if node.parameter("__array__") in ("string", "bytestring"):
            families.add("string")
        elif node.is_record or node.is_union:
            stack.extend(node.contents)
        elif node.is_numpy:
            if node.primitive == "uint64":
                families.add("uint64")
            elif node.primitive.startswith("datetime64"):
                families.add("datetime")
            elif node.primitive.startswith("timedelta64"):
                if node.primitive.endswith(("[M]", "[Y]")):
                    families.add("timedelta-calendar")
                else:
                    families.add("timedelta")
            elif node.primitive.startswith(("float", "complex")):
                families.add("float")
            else:
                families.add("integer")
        elif not node.is_unknown:
            stack.append(node.content)
    return families


def _reaches_empty_union(layout: ak.contents.Content) -> bool:
    """Return `True` if a union that no values reach is in the layout.

    Descends through reachable values only: option and indexed nodes
    are projected, union branches are projected one by one, and
    record, regular, and list contents are trimmed to the range their
    parent references.
    """
    if layout.is_union:
        return layout.length == 0 or any(
            _reaches_empty_union(layout.project(i)) for i in range(len(layout.contents))
        )
    if layout.is_record:
        return any(_reaches_empty_union(c[: layout.length]) for c in layout.contents)
    if layout.is_option or layout.is_indexed:
        return _reaches_empty_union(layout.project())
    # A RegularArray is also a list, so this branch must come first.
    if layout.is_regular:
        return _reaches_empty_union(layout.content[: layout.length * layout.size])
    if layout.is_list:
        lst = layout.to_ListOffsetArray64(False)
        return _reaches_empty_union(lst.content[lst.offsets[0] : lst.offsets[-1]])
    return False
