# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Helpers shared by the operation test modules."""

import inspect
import math
import re
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

import awkward as ak

# The linear datetime64/timedelta64 units, as exact attosecond multiples.
# The calendar units (month, year) are absent: their length varies, so no
# constant factor converts them to a linear unit.
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


def merges_out_of_range_temporal_value(a: ak.Array) -> bool:
    """Return `True` if flattening every level would merge a value out of range.

    `ak.flatten(axis=None)` and `ak.ravel` merge the leaves they
    collect, and merging temporal leaves of one family converts every
    value to the unit the merge takes — the greatest common divisor of
    the steps present. A value of magnitude above `(2**63 - 1) // f`,
    with `f` its leaf's conversion factor, has no `int64`
    representation in that unit, and awkward refuses the merge with
    `ValueError` rather than leave NumPy to raise `OverflowError`
    (2.5.0 and later) or to wrap the value silently (earlier).

    Datetimes and timedeltas are judged apart, as they merge apart.
    Every value in each temporal leaf's buffer counts, reachable or
    not, and a family holding a unit with no constant factor — a
    calendar unit (month, year) or the deprecated generic unit —
    counts whole: both over-match, which costs only coverage. `NaT`
    always converts, so it never counts.
    """
    steps: dict[str, list[tuple[int, np.ndarray]]] = {
        "datetime64": [],
        "timedelta64": [],
    }
    leaf_counts = dict.fromkeys(steps, 0)
    without_factor = set()
    stack = [a.layout]
    while stack:
        node = stack.pop()
        if node.parameter("__array__") in ("string", "bytestring"):
            continue
        if node.is_record or node.is_union:
            stack.extend(node.contents)
        elif node.is_numpy:
            matched = re.fullmatch(
                r"(datetime64|timedelta64)(?:\[(\d*)([a-zA-Z]+)\])?", str(node.dtype)
            )
            if matched is None:
                continue
            kind, multiplier, unit = matched.groups()
            leaf_counts[kind] += 1
            attoseconds = None if unit is None else ATTOSECONDS_PER_TIME_UNIT.get(unit)
            if attoseconds is None:
                without_factor.add(kind)
            else:
                data = np.asarray(node.data).ravel().view(np.int64)
                steps[kind].append((int(multiplier or "1") * attoseconds, data))
        elif not node.is_unknown:
            stack.append(node.content)

    nat = np.iinfo(np.int64).min
    for kind, family in steps.items():
        if leaf_counts[kind] < 2:
            continue
        if kind in without_factor:
            return True
        merged = math.gcd(*(step for step, _ in family))
        for step, data in family:
            factor = step // merged
            if factor == 1:
                continue
            limit = (2**63 - 1) // factor
            values = data[data != nat]
            if values.size and np.abs(values).max() > limit:
                return True
    return False
