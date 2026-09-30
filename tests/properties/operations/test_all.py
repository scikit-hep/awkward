# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Property tests for `ak.all`.

The module follows the template described in the package docstring.
"""

from typing import Any, TypedDict, cast

import hypothesis_awkward.strategies as st_ak
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import awkward as ak
from tests.properties.operations import known_issues, util


class Kwargs(TypedDict, total=False):
    """Options for `ak.all`."""

    axis: int | None
    keepdims: bool
    mask_identity: bool
    highlevel: bool
    behavior: dict[str, Any] | None
    attrs: dict[str, Any] | None


DEFAULTS = Kwargs(
    axis=None,
    keepdims=False,
    mask_identity=False,
    highlevel=True,
    behavior=None,
    attrs=None,
)


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together: `behavior` and `attrs` depend on `highlevel`."""

    highlevel: bool
    behavior: dict[str, Any] | None
    attrs: dict[str, Any] | None


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `ak.all`'s parameters."""
    util.assert_kwargs_match_signature(
        func=ak.all,
        data_param_names={"array"},
        kwargs_cls=Kwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def st_kwargs(draw: st.DrawFn, array: ak.Array) -> Kwargs:
    """Strategy for options of `ak.all` on `array`.

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
                # array; a matching entry can change what raises (a
                # registered record reducer makes records reducible),
                # so `_should_not_raise` must then model the drawn
                # entries.
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
        st.fixed_dictionaries(
            {},
            optional={
                "axis": _st_axes(),
                "keepdims": st.booleans(),
                "mask_identity": st.booleans(),
            },
        )
    )

    return cast(Kwargs, {**related, **optional_independent})


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert `ak.all` does not raise on a draw expected to succeed."""
    a = data.draw(st_ak.constructors.arrays(), label="a")
    kwargs = data.draw(st_kwargs(a), label="kwargs")

    assume(_should_not_raise(a.layout.form, kwargs))

    assume(not _would_raise_from_known_issue(a, kwargs))

    ak.all(a, **kwargs)

    # TODO: assert properties


def _should_not_raise(form: ak.forms.Form, kwargs: Kwargs) -> bool:
    """Return `True` if the operation should be successful.

    Conservative: `False` makes no statement — the call may still
    succeed; a rule shown too permissive by a failure is narrowed
    toward `False`. Only `axis` decides: the other options, as they
    are drawn, never affect whether `ak.all` raises.

    The rules, written against `ak.all`'s deliberate error paths
    (records, irreducible unions, axis beyond depth):

    - A form `ak.all` refuses to run on (see `_reducible`): never
      `True`, whatever the axis.
    - On an accepted form: `axis=None` (reduce everything) always
      succeeds, and an integer axis is `True` when it resolves to
      `0 <= axis < depth`. No level type is excluded: a string is a
      reducible leaf — `ak.all` runs over its characters, and
      `minmax_depth` counts it as one level.
    """
    if not _reducible(form):
        return False
    axis = kwargs.get("axis", DEFAULTS["axis"])
    if axis is None:
        return True
    axis = _normalize_axis(axis, form)
    if axis < 0:
        return False
    return axis < form.minmax_depth[1]


def _would_raise_from_known_issue(a: ak.Array, kwargs: Kwargs) -> bool:
    """Return `True` if an open issue affects the operation.

    One clause per issue; the predicates and their descriptions are
    in `known_issues`, and this function applies any option
    conditions.
    """
    if known_issues.has_issue_4259(a):
        return True

    axis = kwargs.get("axis", DEFAULTS["axis"])

    match axis:
        case int():
            if known_issues.has_issue_4264(a):
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


def _reducible(form: ak.forms.Form) -> bool:
    """Return `True` if `ak.all` runs on this form at every axis it has.

    A record anywhere makes `ak.all` raise at every axis, `None` and
    `0` included; a union raises only in some configurations, and
    deciding them is left out, so both are refused and the accepted
    forms are single chains of list, option, and indexed nodes down
    to a leaf.
    """
    node = form
    while True:
        if node.is_record or node.is_union:
            return False
        if node.is_numpy or node.is_unknown:
            return True
        node = node.content
