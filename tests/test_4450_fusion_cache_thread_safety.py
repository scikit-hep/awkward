# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import inspect
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from awkward._connect.cpu import _fusion_codegen as cfc


def test_concurrent_cold_miss():
    # Use an isolated cache so parallel tests cannot warm or evict this entry.
    compile_op = inspect.unwrap(cfc._compile_op)
    calls = []

    def slow_compile(source):
        calls.append(source)
        # Release the GIL so the regression is observable on ordinary Python too.
        time.sleep(0.05)
        return compile_op(source)

    cached = cfc._serialized_cache(slow_compile)
    start = threading.Barrier(8)

    def lookup():
        start.wait(timeout=10)
        return cached("c[0] * 2 + 1")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: lookup(), range(8)))
    assert len(calls) == 1
    assert all(result is results[0] for result in results)
    assert results[0]([3]) == 7
    assert cached.cache_info().misses == 1
    assert cached.cache_info().hits == 7


def test_bounded_cache_can_evict():
    cached = cfc._serialized_cache(inspect.unwrap(cfc._compile_op))
    first = cached("c[0]")
    assert cached("c[0]") is first
    for i in range(128):
        cached(f"c[0] + {i}")
    assert cached.cache_info().currsize == 128
    assert cached.cache_parameters()["maxsize"] == 128
    assert cached("c[0]") is not first
    cached.cache_clear()
    assert cached.cache_info().currsize == 0
