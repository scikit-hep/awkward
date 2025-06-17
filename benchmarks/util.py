from __future__ import annotations

import functools
import math

import google_benchmark
import numpy as np  # noqa: TID251

import awkward as ak


# reproducible rng
def rng():
    return np.random.default_rng(seed=42)


def benchmark(*args):
    def decorator(func):
        for test_case in args[0]:

            @google_benchmark.register(name=test_case["name"])
            @functools.wraps(func)
            def wrapper(state, test_case=test_case):
                return func(state, **test_case)

        return wrapper

    return decorator


def _generate_counts(sum_upto: int, how_many: int) -> np.ndarray:
    counts = (
        rng()
        .multinomial(sum_upto, rng().dirichlet(np.ones(how_many) * 0.3))
        .astype("int")
    )
    assert np.sum(counts) == sum_upto
    return counts


def Jagged(length, dtype):
    """creates a singely jagged array"""
    flat_content = rng().random(length, dtype)

    # seems like a reasonable heuristic
    powof2 = int(math.log(length) / math.log(2))
    how_many = (1 << (powof2 // 2)) * 10

    assert how_many < length

    counts = _generate_counts(length, how_many)
    return ak.unflatten(flat_content, counts)


def Flat(length, dtype):
    """creates a flat array"""
    return ak.Array(rng().random(length, dtype))


def format_benchmark_name(params: dict) -> str:
    base = "ak." + params.pop("op_name", "??")
    array = params.pop("array", "??")
    length = params.pop("length", "??")
    dtype_short = (
        params.pop("dtype", "??")
        .replace("float", "f")
        .replace("int", "i")
        .replace("complex", "c")
    )

    pretty_name = f"{base}({array}<{dtype_short}[{length}]>"

    # any extra parameters to the function, e.g. `axis=0`
    for k, v in params.items():
        pretty_name += f", {k}={v}"

    pretty_name += ")"
    return pretty_name
