from __future__ import annotations

import google_benchmark
from util import Flat, Jagged, benchmark

import awkward as ak


def _prepare_fun_benchmark(fun):
    return [
        {
            "name": f"ak.{fun.__name__}/array={mkarr.__name__}/{length=}/{dtype=}",
            "mkarr": mkarr,
            "length": length,
            "dtype": dtype,
            "fun": fun,
        }
        for mkarr in (Jagged, Flat)
        for length in [1 << i for i in (12, 16, 20)]
        for dtype in ["float64"]
    ]


def _general_fun_benchmark(state, **kwargs):
    mkarr = kwargs["mkarr"]
    length = kwargs["length"]
    dtype = kwargs["dtype"]
    fun = kwargs["fun"]

    # create singely jagged awkward array
    ak_array = mkarr(length, dtype)

    # run measurement
    while state:
        fun(ak_array)

    # track how many elements per second are processed
    state.counters["elements_per_second"] = google_benchmark.Counter(
        length * state.iterations, google_benchmark.Counter.kIsRate
    )


# extend for more misc funs
FUNS = [
    ak.angle,
    ak.drop_none,
    ak.imag,
    ak.is_none,
    ak.is_valid,
    ak.nan_to_none,
    ak.nan_to_num,
    ak.real,
    ak.round,
    ak.validity_error,
]

# register as benchmarks
for fun in FUNS:

    @benchmark(_prepare_fun_benchmark(fun=fun))
    def _(state, **kwargs):
        _general_fun_benchmark(state, **kwargs)


if __name__ == "__main__":
    google_benchmark.main()
