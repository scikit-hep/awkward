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

import re

import numpy as np

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
    """Return `True` if a `float16`, `float128`, or `complex256` leaf is in the array.

    `ak.sort`, `ak.argsort`, and every reducer except `ak.count`
    raise `KeyError` from the kernel-table lookup when a `float16`
    leaf reaches a kernel (`ak.sum`, `ak.min`, and `ak.max` reach one
    only in a partial reduction; a whole-array reduction takes a
    NumPy path): awkward-cpp has no `float16` kernels, and the dtype
    is neither mapped onto an existing kernel
    (`KernelReducer._dtype_for_kernel` does this for datetimes and
    complex numbers) nor refused with a documented error. `float128`
    and `complex256` fail the same lookup (as `numpy.longdouble` and
    `numpy.clongdouble` in the kernel key) on the platforms where
    NumPy provides them — continuous integration on Linux found both;
    macOS NumPy has neither. Which leaves reach a kernel depends on
    the operation and the layout, so every matching leaf counts here,
    although one that never reaches a kernel works. Reported:
    https://github.com/scikit-hep/awkward/issues/4259
    """
    stack = [a.layout.form]
    while stack:
        node = stack.pop()
        if node.is_numpy:
            if node.primitive in ("float16", "float128", "complex256"):
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
    timedeltas, which NumPy promotes but the merge rejects with
    `TypeError`, are grouped as non-promotable; that divergence is
    why `np.result_type` is not a substitute. A unit pair of one
    family that NumPy itself cannot convert is `has_issue_4278`'s
    trigger, not this one. Unprobed combinations may still fail the
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


def has_issue_4278(a: ak.Array) -> bool:
    """Return `True` if one family's temporal units need an overflowing conversion.

    `ak.flatten(axis=None)`, `ak.ravel`, and `axis=None` reductions
    merge the collected leaves with `ak._do.mergemany`; two
    timedelta or two datetime leaves merge through NumPy's unit
    conversion, which refuses a factor of 2**56 or more (any of the
    factor's top eight bits set) with `OverflowError` ("Integer
    overflow getting a common metadata divisor") — `[as]` with `[s]`
    or coarser, `[fs]` with `[h]` or coarser, and `[ps]` with `[D]`
    or coarser, for timedeltas and datetimes alike, and `[M]` or
    `[Y]` datetimes with `[ps]` or finer. An intermediate unit
    does not change the outcome, and other leaves in the array do
    not prevent the failure (`int64` beside such a pair, a
    promotable mix, still fails). The trigger computes the factor
    between the extreme base units present, per family, exactly in
    attoseconds (`_overflowing_unit_span`); merge failures across
    families are `has_issue_4261`'s. Reported:
    https://github.com/scikit-hep/awkward/issues/4278
    """
    form = a.layout.form
    return _contains_record_or_union(form) and _overflowing_unit_span(form)


def has_issue_4280(a: ak.Array) -> bool:
    """Return `True` if a temporal value is too large for its family's finest unit.

    Merging temporal leaves of one family converts every value to
    the finest unit present (`ak.flatten(axis=None)`, `ak.ravel`,
    and `axis=None` reductions reach the merge); a value of
    magnitude above `(2**63 - 1) // f`, with `f` the conversion
    factor from the value's unit to the finest one, overflows int64,
    and the merge raises `OverflowError` ("Overflow when converting
    between datetime64 units") instead of building a union or
    raising a documented error. The bound is symmetric in sign, and
    `NaT` converts safely. The dtype pair alone stays under
    `has_issue_4278`'s factor bound, so the stored values decide.
    Every value in each temporal leaf's buffer counts, reachable or
    not — an over-match, which is harmless. A datetime calendar unit
    is not counted: NumPy converts calendar values without an
    overflow check, so an out-of-range value wraps silently and no
    exception reports it. NumPy 2.5.0 added the linear-unit value
    check; on earlier versions that cast also wraps silently and the
    operation succeeds, so the skip is an over-match there. Reported:
    https://github.com/scikit-hep/awkward/issues/4280
    """
    form = a.layout.form
    if not _contains_record_or_union(form):
        return False
    leaves: dict[str, list[tuple[int, np.ndarray]]] = {
        "datetime64": [],
        "timedelta64": [],
    }
    stack = [a.layout]
    while stack:
        node = stack.pop()
        if node.parameter("__array__") in ("string", "bytestring"):
            continue
        if node.is_record or node.is_union:
            stack.extend(node.contents)
        elif node.is_numpy:
            matched = re.fullmatch(
                r"(datetime64|timedelta64)\[(\d*)([a-zA-Z]+)\]", str(node.dtype)
            )
            if matched is None:
                continue
            kind, multiplier, unit = matched.groups()
            attos = _ATTOSECONDS_PER_UNIT.get(unit)
            if attos is not None:
                data = np.asarray(node.data).ravel().view(np.int64)
                leaves[kind].append((int(multiplier or "1") * attos, data))
        elif not node.is_unknown:
            stack.append(node.content)
    nat = np.iinfo(np.int64).min
    for family in leaves.values():
        if len(family) < 2:
            continue
        finest = min(attos for attos, _ in family)
        for attos, data in family:
            factor = attos // finest
            if factor == 1:
                continue
            limit = (2**63 - 1) // factor
            values = data[data != nat]
            if values.size and np.abs(values).max() > limit:
                return True
    return False


def has_issue_4282(a: ak.Array) -> bool:
    """Return `True` if a regular list with no variable-length dimension below it is under a union.

    To restore the interleaving order of a union,
    `UnionArray._remove_structure` broadcasts each branch against its
    rows' positions (`ak.flatten(axis=None)`, `ak.ravel`, and
    `axis=None` reductions reach it), which relies on the
    replicate-per-row semantics of variable-length broadcasting; a
    branch with only regular dimensions broadcasts by NumPy's
    right-aligned rules instead, treating the flat positions as one
    inner vector. An innermost regular size of 1 raises `IndexError`
    ("cannot slice NumpyArray ...") from a position marker longer
    than the values, another innermost size differing from the
    branch's row count raises `ValueError` ("cannot broadcast
    RegularArray ..."), and an innermost size equal to the row count
    succeeds with a silently wrong order, which a no-raise test does
    not observe. The trigger is a regular list with no
    variable-length list anywhere below it, itself anywhere under a
    union node, descending through list, option, indexed, and
    record nodes; a regular list of variable-length lists is
    `has_issue_4262`'s trigger. Which configurations reach the
    broadcast depends on the current implementation's internals, so
    every such regular list counts here, although multi-field-record
    or string content, or a single-row or innermost-size-0 branch,
    happens to work (a single-field record still flattens to one
    leaf and stays on the affected path).
    Reported: https://github.com/scikit-hep/awkward/issues/4282
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
            if (
                in_union
                and node.is_regular
                and not _contains_variable_length_list(node.content)
            ):
                return True
            stack.append((node.content, in_union))
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


def _contains_variable_length_list(form: ak.forms.Form) -> bool:
    """Return `True` if a `var`-type list is somewhere in the tree.

    A string node is a leaf, not a list: the descent stops there.
    """
    stack = [form]
    while stack:
        node = stack.pop()
        if node.parameter("__array__") in ("string", "bytestring"):
            continue
        if node.is_list and not node.is_regular:
            return True
        if node.is_record or node.is_union:
            stack.extend(node.contents)
        elif not (node.is_numpy or node.is_unknown):
            stack.append(node.content)
    return False


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


# The linear datetime units, as exact attosecond multiples; the
# calendar units month and year are separate, because only datetimes
# convert with them.
_ATTOSECONDS_PER_UNIT = {
    "as": 1,
    "fs": 10**3,
    "ps": 10**6,
    "ns": 10**9,
    "us": 10**12,
    "ms": 10**15,
    "s": 10**18,
    "m": 60 * 10**18,
    "h": 3600 * 10**18,
    "D": 86400 * 10**18,
    "W": 604800 * 10**18,
}

# A datetime — not a timedelta — converts with the calendar units
# month and year; their average Gregorian lengths, as exact
# attosecond multiples, reproduce NumPy's accept/reject boundary at
# every probed unit pair.
_DATETIME_CALENDAR_ATTOSECONDS = {
    "M": 2629746 * 10**18,
    "Y": 31556952 * 10**18,
}

# NumPy's `get_datetime_units_factor` treats a unit-conversion factor
# with any of its top eight bits set as an overflow, so the bound is
# 2**56, far below `int64`'s 2**63 - 1. The factor multiplies base
# units only: a unit multiplier (as in `datetime64[10s]`) never
# enters it.
_UNIT_FACTOR_BOUND = 2**56


def _overflowing_unit_span(form: ak.forms.Form) -> bool:
    """Return `True` if one family's extreme temporal units would fail to convert.

    Judged per family — datetimes and timedeltas separately, matching
    the grouping in `_leaf_families` — although a form containing
    both datetimes and timedeltas already matches the family check in
    `has_issue_4261`, so the split never decides. The factor between
    the extreme base units present is computed exactly in attoseconds
    and compared against `_UNIT_FACTOR_BOUND`; a unit multiplier is
    ignored, as NumPy's check ignores it. A timedelta calendar unit
    (month, year) does not convert to a linear unit and is left to
    the family grouping; a datetime calendar unit is counted at its
    average length (`_DATETIME_CALENDAR_ATTOSECONDS`), which
    reproduces the observed boundary. A string's character content
    is not walked.
    """
    spans: dict[str, set[int]] = {"datetime64": set(), "timedelta64": set()}
    stack = [form]
    while stack:
        node = stack.pop()
        if node.parameter("__array__") in ("string", "bytestring"):
            continue
        if node.is_record or node.is_union:
            stack.extend(node.contents)
        elif node.is_numpy:
            matched = re.fullmatch(
                r"(datetime64|timedelta64)\[\d*([a-zA-Z]+)\]", node.primitive
            )
            if matched is None:
                continue
            kind, unit = matched.groups()
            attos = _ATTOSECONDS_PER_UNIT.get(unit)
            if attos is None and kind == "datetime64":
                attos = _DATETIME_CALENDAR_ATTOSECONDS.get(unit)
            if attos is not None:
                spans[kind].add(attos)
        elif not node.is_unknown:
            stack.append(node.content)
    return any(
        max(units) // min(units) >= _UNIT_FACTOR_BOUND
        for units in spans.values()
        if units
    )
