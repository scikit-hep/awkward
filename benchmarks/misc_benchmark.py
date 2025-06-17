from __future__ import annotations

import google_benchmark
from util import Jagged, benchmark, format_benchmark_name

import awkward as ak


def _prepare_fun_benchmark(fun):
    return [
        {
            "name": format_benchmark_name(
                {
                    "op_name": fun.__name__,
                    "array": mkarr.__name__,
                    "length": length,
                    "dtype": dtype,
                }
            ),
            "mkarr": mkarr,
            "length": length,
            "dtype": dtype,
            "fun": fun,
        }
        for mkarr in [Jagged]
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

    # for ak.imag/real we need to add imaginary component
    if fun in (ak.imag, ak.real):
        ak_array = ak_array + 1j * ak_array

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
