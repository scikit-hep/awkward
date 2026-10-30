# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Predicates for known issues; the property tests skip affected draws.

Each issue has one public predicate, named `has_issue_<number>` after
the issue it detects, with the description and link in its docstring;
an issue not yet reported takes a placeholder number (`NNN1`, `NNN2`,
...). A predicate takes only the array — nothing else decides whether
an array is affected — so any operation's test module can use it. A
test module calls the predicates it needs from one function of its
own (`_would_raise_from_known_issue` in the operation modules,
`has_issues` in the roundtrip modules) and applies any option
conditions there (a negative `axis` for `ak.flatten`, for example). A
predicate may over-match — an extra skipped draw is harmless, and a
missed one fails the tests — so widen a trigger rather than narrow
it. Delete a predicate in the pull request that fixes its issue: the
skipped draws are then tested again. If an issue is resolved as
intended behavior instead, the exclusion moves into the design
predicate (`_should_not_raise`) of each affected operation.
"""

import json
import re

import numpy as np

import awkward as ak


def has_issue_4221(a: ak.Array) -> bool:
    """Return `True` if a non-empty `IndexedOptionArray` has empty content.

    `to_arrow` crashes projecting nulls over a zero-length content
    (`IndexError` in `ListOffsetArray._to_arrow`). Reported:
    https://github.com/scikit-hep/awkward/issues/4221
    """
    stack = [a.layout]
    while stack:
        node = stack.pop()
        if (
            isinstance(node, ak.contents.IndexedOptionArray)
            and node.length > 0
            and node.content.length == 0
        ):
            return True
        stack.extend(_children(node))
    return False


def has_issue_4228(a: ak.Array) -> bool:
    """Return `True` if a union's option-type children or incoming validity break `to_arrow`.

    `UnionArray._to_arrow` converts each child whole but sizes it by
    the tag's highest index plus one, so an option-type child holding
    more nulls than that length fails in `pyarrow.Array.from_buffers`
    (`ArrowInvalid`, "Null count exceeds array length"), a masked child
    longer than that length fails in `ListOffsetArray._to_arrow` when a
    list or string lies below the option (`IndexError` or `ValueError`,
    with or without nulls; an `UnmaskedArray` passes no validity bytes
    and is exempt), and it scatters incoming validity bytes with a
    per-child index that it misapplies when the index skips values
    (`IndexError`). Any operation that drops union elements without
    trimming the children creates the first two conditions from a valid
    union.

    Each node on the stack carries two flags. `nullable` tracks whether
    enclosing option validity reaches the node. `sliced` tracks whether
    `to_arrow` reaches the node through a transformation that can drop
    union elements. A list or regular node counts only when it does not
    span its whole content, and a record field only when it is longer
    than the record. Under such a node any null in a masked child of a
    union, or any masked child with a list below, triggers; at an
    unsliced union the counts and lengths are compared. Reported:
    https://github.com/scikit-hep/awkward/issues/4228
    """
    stack = [(a.layout, False, False)]
    while stack:
        node, nullable, sliced = stack.pop()
        if node.is_union:
            if nullable:
                return True
            tags = node.tags.data
            index = node.index.data[: len(tags)]
            for tag, x in enumerate(node.contents):
                if not x.is_option or isinstance(x, ak.contents.UnmaskedArray):
                    continue
                masked_list = _contains_list(x.content)
                if _null_count(x) == 0 and not masked_list:
                    continue
                if sliced:
                    return True
                used = index[tags == tag]
                length = int(used.max()) + 1 if len(used) else 0
                if _null_count(x) > length or (masked_list and x.length > length):
                    return True
            stack.extend((x, False, True) for x in node.contents)
        elif node.is_record:
            stack.extend(
                (x, nullable, sliced or x.length > node.length) for x in node.contents
            )
        elif node.is_regular:
            spans = node.length * node.size == node.content.length
            stack.append((node.content, False, sliced or not spans))
        elif node.is_option:
            indexed = isinstance(node, ak.contents.IndexedOptionArray)
            stack.append((node.content, True, sliced or indexed))
        elif node.is_indexed:
            stack.append((node.content, nullable, True))
        elif node.is_list:
            if isinstance(node, ak.contents.ListOffsetArray):
                spans = bool(
                    node.offsets[0] == 0 and node.offsets[-1] == node.content.length
                )
                child_sliced = sliced or not spans
            else:
                child_sliced = True
            stack.append((node.content, False, child_sliced))
    return False


def has_issue_4229(a: ak.Array) -> bool:
    """Return `True` if a regular list of size 0 is in the array.

    `from_arrow` (and `from_feather`) loses the length of a size-0
    fixed-size list: a `3 * 0 * float64` array reads back with length 0.
    Reported: https://github.com/scikit-hep/awkward/issues/4229
    """
    stack = [a.layout]
    while stack:
        node = stack.pop()
        if node.is_regular and node.size == 0:
            return True
        stack.extend(_children(node))
    return False


def has_issue_4241(a: ak.Array) -> bool:
    """Return `True` if a written float value reads back changed from JSON.

    `from_json` parses numbers with rapidjson's approximate default
    algorithm, not `kParseFullPrecisionFlag`: a shortest-repr double can
    read back off by one ULP, and re-parsing its new repr can drift
    further, so affected values neither roundtrip nor reach a fixed
    point (silent corruption).

    Only values the writer visits count, as in `_has_infinity` in
    `test_to_from_json`: the walk descends into projected options and
    indexed nodes, union children by projection, and the spanned part
    of record fields and list contents. Whether a value is affected
    depends on its digit string, so each float leaf is checked directly
    against a `from_json` readback. Reported:
    https://github.com/scikit-hep/awkward/issues/4241
    """
    stack = [a.layout]
    while stack:
        node = stack.pop()
        if node.is_numpy:
            if node.dtype.kind != "f":
                continue
            # `to_json` serializes the repr of the exact widened double
            data = np.asarray(node.data, dtype=np.float64)
            data = data[np.isfinite(data)]
            if data.size == 0:
                continue
            returned = np.asarray(ak.from_json(json.dumps(data.tolist())))
            if not np.array_equal(returned, data):
                return True
        elif node.is_union:
            stack.extend(node.project(i) for i in range(len(node.contents)))
        elif node.is_record:
            stack.extend(c[: node.length] for c in node.contents)
        elif node.is_option or node.is_indexed:
            stack.append(node.project())
        elif node.is_regular:
            stack.append(node.content[: node.length * node.size])
        elif node.is_list:
            node = node.to_ListOffsetArray64(False)
            stack.append(node.content[node.offsets[0] : node.offsets[-1]])
    return False


def has_issue_4255(a: ak.Array) -> bool:
    """Return `True` if an indexed node over unknown type receives option validity.

    `to_arrow` writes an indexed node over unknown-type content below
    an enclosing option as a nullable Arrow null type without the
    revertable stamp, so `from_arrow` cannot strip the option the field
    metadata says it must (`AttributeError` in `remove_optiontype`).
    Each node on the stack carries `nullable`, which tracks whether
    enclosing option validity reaches the node. Reported:
    https://github.com/scikit-hep/awkward/issues/4255
    """
    stack = [(a.layout, False)]
    while stack:
        node, nullable = stack.pop()
        if node.is_union:
            stack.extend((x, False) for x in node.contents)
        elif node.is_record:
            stack.extend((x, nullable) for x in node.contents)
        elif node.is_regular:
            stack.append((node.content, False))
        elif node.is_option:
            stack.append((node.content, True))
        elif node.is_indexed:
            if nullable and node.content.is_unknown:
                return True
            stack.append((node.content, nullable))
        elif node.is_list:
            stack.append((node.content, False))
    return False


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


def has_issue_4274(a: ak.Array) -> bool:
    """Return `True` if a root `UnmaskedArray` over a record has a uint32-indexed field.

    `to_arrow_table` (which `to_feather` and `to_parquet` call) projects
    each record field through a root `UnmaskedArray` with
    `to_IndexedOptionArray64`, which passes an `IndexedArray`'s uint32
    index into `IndexedOptionArray` unconverted (`TypeError`). Only the
    root takes this path: nested layouts convert through `_to_arrow`,
    which projects the indexed node away. Reported:
    https://github.com/scikit-hep/awkward/issues/4274
    """
    layout = a.layout
    if not (isinstance(layout, ak.contents.UnmaskedArray) and layout.content.is_record):
        return False
    return any(
        isinstance(x, ak.contents.IndexedArray) and x.index.dtype == np.dtype(np.uint32)
        for x in layout.content.contents
    )


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


def has_issue_4316(a: ak.Array) -> bool:
    """Return `True` if a union's list child holds records or lists.

    `ak.array_equal`, the roundtrip tests' oracle, returns `False` for
    two equal unions when one side's index into a list child is a
    permutation of the other's and the list's content is a record or a
    list: the visitor in `ak_almost_equal` carries the used child
    lazily, the list branch then compacts the permuted side with another
    lazy carry, which wraps a record in an `IndexedArray` and turns a
    list into a `ListArray`, and the `same_content_types` class check
    fails before the projection that would undo it. The roundtrip pairs
    such sides whenever `to_packed` renumbers the original's index (an
    over-long child, or a union that was sliced) while the
    reconstruction keeps the drawn permutation. A regular list child
    compares equal through the roundtrip but not in general
    (`1 * var * {}`), so it is excluded too. Reported:
    https://github.com/scikit-hep/awkward/issues/4316
    """
    stack = [a.layout]
    while stack:
        node = stack.pop()
        if node.is_union and any(
            x.is_list and _carried_lazily(x.content) for x in node.contents
        ):
            return True
        stack.extend(_children(node))
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


def _children(layout: ak.contents.Content) -> list[ak.contents.Content]:
    """Return the direct child layouts of a node."""
    if layout.is_record or layout.is_union:
        return list(layout.contents)
    if layout.is_option or layout.is_indexed or layout.is_list:
        return [layout.content]
    return []


def _null_count(layout: ak.contents.Content) -> int:
    """Return the number of missing values in an option node."""
    return int(np.count_nonzero(layout.mask_as_bool(valid_when=False)))


def _contains_list(layout: ak.contents.Content) -> bool:
    """Return `True` if a list node (a string or regular list included)
    is the node or lies below it."""
    if layout.is_list:
        return True
    return any(_contains_list(x) for x in _children(layout))


def _carried_lazily(layout: ak.contents.Content) -> bool:
    """Return `True` if a lazy carry changes the node's class: a record
    becomes an `IndexedArray` and a list a `ListArray`. Indexed and
    option wrappers are looked through because `to_packed` projects
    them."""
    while layout.is_indexed or layout.is_option:
        layout = layout.content
    return layout.is_list or layout.is_record


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
