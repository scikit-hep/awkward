# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Property tests for `ak.ravel`.

The module follows the template described in the package docstring.
"""

from typing import Any, TypedDict, cast

import hypothesis_awkward.strategies as st_ak
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import awkward as ak
from tests.properties.operations import known_issues, util


class Kwargs(TypedDict, total=False):
    """Options for `ak.ravel`."""

    highlevel: bool
    behavior: dict[str, Any] | None
    attrs: dict[str, Any] | None


DEFAULTS = Kwargs(highlevel=True, behavior=None, attrs=None)


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together: `behavior` and `attrs` depend on `highlevel`."""

    highlevel: bool
    behavior: dict[str, Any] | None
    attrs: dict[str, Any] | None


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `ak.ravel`'s parameters."""
    util.assert_kwargs_match_signature(
        func=ak.ravel,
        data_param_names={"array"},
        kwargs_cls=Kwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def st_kwargs(draw: st.DrawFn, array: ak.Array) -> Kwargs:
    """Strategy for options of `ak.ravel` on `array`.

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
                # array; `ak.ravel` reads `behavior` only when
                # wrapping its output, without affecting what raises.
                ret["behavior"] = draw(st_ak.none_or(st.builds(dict)))
            if draw(st.booleans()):
                # TODO: generate non-empty `attrs`. Any value is inert
                # except under the reserved `__named_axis__` key
                # (`awkward._namedaxis.NAMED_AXIS_KEY`), which holds
                # the named-axis mapping: generate that key only with
                # a valid mapping, or not at all.
                ret["attrs"] = draw(st_ak.none_or(st.builds(dict)))
        return ret

    related = draw(_st_related_kwargs())

    return cast(Kwargs, dict(related))


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert `ak.ravel` does not raise on a draw expected to succeed."""
    a = data.draw(st_ak.constructors.arrays(), label="a")
    kwargs = data.draw(st_kwargs(a), label="kwargs")

    assume(_should_not_raise(a.layout.form, kwargs))

    assume(not _would_raise_from_known_issue(a, kwargs))

    ak.ravel(a, **kwargs)

    # TODO: assert properties


def _should_not_raise(form: ak.forms.Form, kwargs: Kwargs) -> bool:
    """Return `True` if the operation should be successful.

    Conservative: `False` makes no statement — the call may still
    succeed; a rule shown too permissive by a failure is narrowed
    toward `False`. `ak.ravel` has no rule to state: it accepts every
    array — its deliberate rejections (a scalar primitive, an
    `ak.Record`) are not arrays — and no option affects whether it
    raises, so every draw is expected to succeed.
    """
    return True


def _would_raise_from_known_issue(a: ak.Array, kwargs: Kwargs) -> bool:
    """Return `True` if an open issue affects the operation.

    One clause per issue; the predicates and their descriptions are
    in `known_issues`, and this function applies any option
    conditions — for `ak.ravel`, none.
    """
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
    if known_issues.has_issue_4283(a):
        return True
    return False
