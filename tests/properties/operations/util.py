# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Helpers shared by the operation test modules."""

import inspect
import re
from collections.abc import Callable, Mapping
from typing import Any

import awkward as ak

# The linear datetime64/timedelta64 units, in attoseconds.
ATTOSECONDS_PER_TIME_UNIT = {
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

# Average Gregorian lengths; only datetimes convert with these.
DATETIME_CALENDAR_ATTOSECONDS = {
    "M": 2629746 * 10**18,
    "Y": 31556952 * 10**18,
}

# NumPy rejects a unit-conversion factor with any of its top eight
# bits set; a unit multiplier never enters the factor.
UNIT_FACTOR_BOUND = 2**56


def assert_kwargs_match_signature(
    func: Callable[..., Any],
    data_param_names: set[str],
    kwargs_cls: type,
    defaults: Mapping[str, Any],
    related_cls: type,
) -> None:
    """Raise unless the option declarations agree with `func`'s parameters.

    `kwargs_cls` and `related_cls` are the module's `TypedDict`
    classes and `defaults` its `DEFAULTS`. Asserts that `kwargs_cls`
    has exactly the parameters of `func` other than
    `data_param_names`, that `defaults` equals the signature's
    default values, and that `related_cls`'s keys are a subset of
    `kwargs_cls`'s.
    """
    option_params = {
        name: p.default
        for name, p in inspect.signature(func).parameters.items()
        if name not in data_param_names
    }
    keys = kwargs_cls.__required_keys__ | kwargs_cls.__optional_keys__
    assert keys == set(option_params)

    assert defaults == option_params

    related_keys = related_cls.__required_keys__ | related_cls.__optional_keys__
    assert related_keys <= keys


def merges_unconvertible_temporal_units(form: ak.forms.Form) -> bool:
    """Return `True` if flattening every level would merge units with no common unit.

    `ak.flatten(axis=None)` and `ak.ravel` merge the leaves they
    collect, and temporal leaves of one family merge through NumPy's
    unit conversion, which refuses a factor of `UNIT_FACTOR_BOUND` or
    more — `[as]` with `[s]` or coarser, `[fs]` with `[h]` or coarser,
    `[ps]` with `[D]` or coarser, and `[M]` or `[Y]` datetimes with
    `[ps]` or finer. Awkward refuses such a merge with `ValueError`
    rather than leave NumPy to raise `OverflowError`.

    Judged per family — datetimes and timedeltas separately — from
    the factor between the extreme base units present, computed
    exactly in attoseconds; a unit multiplier is ignored, as NumPy's
    check ignores it. A timedelta calendar unit, and a form holding
    both families, fail in `known_issues.has_issue_4261` first, so
    matching them here is an over-match. A string's character content
    is not walked, and leaves that no value reaches count too.
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
            attos = ATTOSECONDS_PER_TIME_UNIT.get(unit)
            if attos is None and kind == "datetime64":
                attos = DATETIME_CALENDAR_ATTOSECONDS.get(unit)
            if attos is not None:
                spans[kind].add(attos)
        elif not node.is_unknown:
            stack.append(node.content)
    return any(
        max(units) // min(units) >= UNIT_FACTOR_BOUND
        for units in spans.values()
        if units
    )
