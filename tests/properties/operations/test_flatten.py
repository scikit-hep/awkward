# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Property tests for `ak.flatten`.

The module follows the template described in the package docstring.
"""

from typing import Any, TypedDict, cast

import hypothesis_awkward.strategies as st_ak
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import awkward as ak
from tests.properties.operations import known_issues, util


class Kwargs(TypedDict, total=False):
    """Options for `ak.flatten`."""

    axis: int | None
    highlevel: bool
    behavior: dict[str, Any] | None
    attrs: dict[str, Any] | None


DEFAULTS = Kwargs(axis=1, highlevel=True, behavior=None, attrs=None)


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together: `behavior` and `attrs` depend on `highlevel`."""

    highlevel: bool
    behavior: dict[str, Any] | None
    attrs: dict[str, Any] | None


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `ak.flatten`'s parameters."""
    util.assert_kwargs_match_signature(
        func=ak.flatten,
        data_param_names={"array"},
        kwargs_cls=Kwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def st_kwargs(draw: st.DrawFn, array: ak.Array) -> Kwargs:
    """Strategy for options of `ak.flatten` on `array`.

    Every option is optional in the drawn dict, so the defaults are
    exercised as well; whether each option appears is drawn separately
    from its value. `behavior` and `attrs` affect only a high-level
    output, so they are drawn only when `highlevel` is not `False`.
    Only `None` and an empty `dict` are drawn for `behavior` (a
    mapping from names to classes) and `attrs`.
    """

    @st.composite
    def _st_related_kwargs(draw: st.DrawFn) -> RelatedKwargs:
        """Strategy for `highlevel` and the options that depend on it."""
        ret = RelatedKwargs()
        if draw(st.booleans()):
            ret["highlevel"] = draw(st.booleans())

        if ret.get("highlevel", DEFAULTS["highlevel"]) is not False:
            if draw(st.booleans()):
                # TODO: generate non-empty `behavior`. An entry is
                # inert unless its key matches a lookup for the drawn
                # array; `ak.flatten`'s observed lookups select only
                # output classes, without affecting what raises.
                ret["behavior"] = draw(st_ak.none_or(st.builds(dict)))
            if draw(st.booleans()):
                # TODO: generate non-empty `attrs`. Any value is inert
                # except under the reserved `__named_axis__` key
                # (`awkward._namedaxis.NAMED_AXIS_KEY`), which holds
                # the named-axis mapping: generate that key only with
                # a valid mapping, or not at all.
                ret["attrs"] = draw(st_ak.none_or(st.builds(dict)))
        return ret

    def _st_axes() -> st.SearchStrategy[int | None]:
        """Strategy for the option `axis` on `array`.

        Draws `None` or an integer in `[-depth, depth)` for the
        array's maximum depth; an axis that does not exist for the
        drawn form is left to `_should_not_raise`.
        """
        depth = array.layout.minmax_depth[1]
        return st_ak.none_or(st.integers(min_value=-depth, max_value=depth - 1))

    related = draw(_st_related_kwargs())

    optional_independent = draw(
        st.fixed_dictionaries({}, optional={"axis": _st_axes()})
    )

    return cast(Kwargs, {**related, **optional_independent})


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert `ak.flatten` does not raise on a draw expected to succeed."""
    a = data.draw(st_ak.constructors.arrays(), label="a")
    kwargs = data.draw(st_kwargs(a), label="kwargs")

    assume(_should_not_raise(a.layout.form, kwargs))

    assume(not _would_raise_from_known_issue(a, kwargs))

    ak.flatten(a, **kwargs)

    # TODO: assert properties


def _should_not_raise(form: ak.forms.Form, kwargs: Kwargs) -> bool:
    """Return `True` if the operation should be successful.

    Conservative: `False` makes no statement — the call may still
    succeed; a rule shown too permissive by a failure is narrowed
    toward `False`. Only `axis` decides: the other options never
    affect whether `ak.flatten` raises.

    The rules, written against `ak.flatten`'s deliberate error paths
    (axis beyond depth, strings, records):

    - `axis=None`: always `True`.
    - Any other axis is first resolved against the depth (see
      `_normalize_axis`); an axis that stays negative is `False`.
    - A resolved `0`: always `True`.
    - A resolved positive axis: `True` when plain lists occupy every
      level of the form down to it (see `_all_lists_down_to`). The
      descent stops at a record or a union, although `ak.flatten`
      also flattens a union whose branches are all lists at that
      level, and the fields of a record above the axis.
    """
    axis = kwargs.get("axis", DEFAULTS["axis"])
    if axis is None:
        return True
    axis = _normalize_axis(axis, form)
    if axis == 0:
        return True
    if axis < 0:
        return False
    return _all_lists_down_to(axis, form)


def _would_raise_from_known_issue(a: ak.Array, kwargs: Kwargs) -> bool:
    """Return `True` if an open issue affects the operation.

    One clause per issue; the predicates and their descriptions are
    in `known_issues`, and this function applies any option
    conditions.
    """
    axis = kwargs.get("axis", DEFAULTS["axis"])

    match axis:
        case None:
            if known_issues.has_issue_4261(a):
                return True
            if known_issues.has_issue_4262(a):
                return True
            if known_issues.has_issue_4278(a):
                return True
            if known_issues.has_issue_4280(a):
                return True
            if known_issues.has_issue_4282(a):
                return True
        case int() if axis < 0:
            if known_issues.has_issue_4260(a):
                return True
    return False


def _normalize_axis(axis: int, form: ak.forms.Form) -> int:
    """Resolve a negative `axis` against the form's depth.

    A non-negative `axis` is already resolved and passes through. A
    resolved axis is non-negative, so a negative result indicates that
    none exists: a negative axis is ambiguous on a branching form
    (the axis is returned unchanged), and an axis below `-depth` is
    beyond the form (the resolution stays negative).
    """
    if axis >= 0:
        return axis
    branches, depth = form.branch_depth
    if branches:
        return axis
    return axis + depth


def _all_lists_down_to(axis: int, form: ak.forms.Form) -> bool:
    """Return `True` if plain lists occupy every level down to `axis`.

    Descends only through option, indexed, and list nodes (regular
    lists included), none of which may carry an `__array__` parameter
    (that parameter overrides structural semantics: a string is a
    list of characters only structurally); a record, union, or leaf
    ends the descent. `axis` must be positive.
    """
    node = form
    levels = 0
    while levels < axis:
        if node.parameter("__array__") is not None:
            return False
        if node.is_option or node.is_indexed:
            node = node.content
        elif node.is_list:
            node = node.content
            levels += 1
        else:
            return False
    return True
