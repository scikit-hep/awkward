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


def has_issue_4264(a: ak.Array) -> bool:
    """Return `True` if an option or a list sits under an untrimmed regular list.

    The caller applies the option condition (for `ak.all`, an integer
    `axis`). A `RegularArray` may validly carry content longer than
    `size * length`, and a node reached by a reduction sizes its carry
    from all of its own rows, ignoring which of them the caller's
    offsets select, while the kernel fills only the entries those
    offsets span, leaving the tail uninitialized. An option directly
    above a list fails a bare
    `AssertionError` in `RegularArray._reduce_next` on every run;
    otherwise the leftover memory decides, and an out-of-range value
    raises `IndexError` from `NumpyArray._carry` (`ak.all`, `ak.any`,
    `ak.sum`, `ak.count`, and `ak.min` are all affected). With content
    trimmed to `size * length` the same reductions succeed.

    A list in the untrimmed content triggers it with no option node
    present, which the issue's title and reproducers do not cover; the
    2026-08-14 nightly drew `1 * 1 * 2 * string`, whose inner
    `RegularArray` addresses two of its `ListOffsetArray`'s three rows.
    Strings are lists, so they are not skipped here. A bare leaf in the
    untrimmed content is unaffected, the reduction then reading no
    carry. Reported:
    https://github.com/scikit-hep/awkward/issues/4264
    """
    stack = [(a.layout, False)]
    while stack:
        node, in_untrimmed = stack.pop()
        if node.is_numpy or node.is_unknown:
            continue
        if (node.is_option or node.is_list) and in_untrimmed:
            return True
        if node.is_regular and node.content.length > node.size * node.length:
            in_untrimmed = True
        if node.is_record or node.is_union:
            stack.extend((c, in_untrimmed) for c in node.contents)
        else:
            stack.append((node.content, in_untrimmed))
    return False


def has_issue_4283(a: ak.Array) -> bool:
    """Return `True` if a record sits directly under an option node.

    `ak.ravel` removes structure with `drop_nones=False`, and an
    option node then keeps itself whole when its content does not
    branch and has depth 1 — a test meant to identify a leaf, which
    a record of leaves also passes. The record then survives into
    the merge inside the kept option, and a leaf part beside it
    makes `ak._do.mergemany` fail, depending on the first part's
    type, with `AssertionError` ("cannot merge NumpyArray with
    RecordArray"; with the record part first, "cannot merge
    RecordArray with NumpyArray") or `ValueError` ("cannot merge
    ListArray with RecordArray."). With only record parts the merge succeeds and the record
    layer survives the flattening, which a no-raise test does not
    observe. `ak.flatten(axis=None)` projects options away
    (`drop_nones=True`) and is unaffected, so `test_flatten` does
    not use this predicate. The trigger is a record as the direct
    content of an option node, anywhere in the form (an option
    cannot hold an indexed node, so nothing sits between); an
    option above a list is descended and its records split into
    their fields. Every such record counts here, although one with
    a non-leaf field gives the option a depth above 1, so the
    option projects and the call happens to work.
    Reported: https://github.com/scikit-hep/awkward/issues/4283
    """
    stack = [a.layout.form]
    while stack:
        node = stack.pop()
        if node.parameter("__array__") in ("string", "bytestring"):
            continue
        if node.is_option:
            if node.content.is_record:
                return True
            stack.append(node.content)
        elif node.is_record or node.is_union:
            stack.extend(node.contents)
        elif node.is_numpy or node.is_unknown:
            continue
        else:
            stack.append(node.content)
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
