import awkward as ak
import google_benchmark

from util import benchmark, Jagged, Flat
import time


def _prepare_reducer_benchmark(reducer):
    return [
        {
            "name": f"ak.{reducer.__name__}/array={mkarr.__name__}/{length=}/{dtype=}/{axis=}",
            "mkarr": mkarr,
            "length": length,
            "dtype": dtype,
            "reducer": reducer,
            "axis": axis,
        }
        for mkarr in (Jagged, Flat)
        for length in [1 << i for i in (12, 16, 20)]
        for dtype in ["float64"]
        for axis in ([None, 0, 1] if mkarr is Jagged else [None])
    ]


def _general_reducer_benchmark(state, **kwargs):
    mkarr = kwargs["mkarr"]
    length = kwargs["length"]
    dtype = kwargs["dtype"]
    reducer = kwargs["reducer"]
    axis = kwargs["axis"]

    # create singely jagged awkward array
    ak_array = mkarr(length, dtype)

    # run measurement
    while state:
        time.sleep(0.1)
        reducer(ak_array, axis=axis)

    # track how many elements per second are processed
    state.counters["elements_per_second"] = google_benchmark.Counter(
        length * state.iterations, google_benchmark.Counter.kIsRate
    )


# extend for more reducers
REDUCERS = [
    ak.all,
    ak.any,
    ak.argmax,
    ak.argmin,
    ak.max,
    ak.mean,
    ak.min,
    ak.prod,
    ak.std,
    ak.sum,
    ak.var,
]

# register as benchmarks
for reducer in REDUCERS:

    @benchmark(_prepare_reducer_benchmark(reducer=reducer))
    def _(state, **kwargs):
        _general_reducer_benchmark(state, **kwargs)


if __name__ == "__main__":
    google_benchmark.main()
