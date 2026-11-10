# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import importlib.util
from pathlib import Path

import pytest

import awkward as ak

# These tests mutate environment/module state and share fixture paths.
pytestmark = pytest.mark.thread_unsafe


@pytest.fixture
def cache():
    path = Path(ak.__file__).parent / "_connect/cuda/_compute_cache.py"
    spec = importlib.util.spec_from_file_location(
        "awkward._connect.cuda._compute_cache", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def configured(cache, monkeypatch, tmp_path):
    monkeypatch.setenv("AWKWARD_CUDA_CACHE_DIR", str(tmp_path))
    path = tmp_path / "cache" / "artifact.bin"
    monkeypatch.setattr(cache, "_cache_path", lambda *args: path)
    marker = object()

    def deserialize(blob):
        if blob != b"compiled":
            raise ValueError("bad artifact")
        return marker

    monkeypatch.setattr(
        cache, "_serialization_api", lambda: (lambda x: b"compiled", deserialize)
    )
    return cache, path, marker


def invoke(cache, factory):
    return cache.load_or_build(factory, source_path=__file__, d_in=1, d_out=2, op=3)


def test_disabled_cache_does_not_inspect_files_or_api(cache, monkeypatch):
    monkeypatch.delenv("AWKWARD_CUDA_CACHE_DIR", raising=False)

    def forbidden():
        raise AssertionError("disabled cache must not inspect serialization")

    monkeypatch.setattr(cache, "_serialization_api", forbidden)
    assert invoke(cache, lambda **kwargs: kwargs) == {"d_in": 1, "d_out": 2, "op": 3}


def test_miss_writes_and_hit_skips_factory(configured):
    cache, path, restored = configured
    original = object()
    assert invoke(cache, lambda **kwargs: original) is original
    assert path.read_bytes() == b"compiled"

    def forbidden(**kwargs):
        raise AssertionError("a hit must not build")

    assert invoke(cache, forbidden) is restored


def test_corrupt_blob_is_rebuilt_and_repaired(configured):
    cache, path, restored = configured
    path.parent.mkdir()
    path.write_bytes(b"truncated")
    original = object()
    assert invoke(cache, lambda **kwargs: original) is original
    assert path.read_bytes() == b"compiled"
    assert invoke(cache, lambda **kwargs: None) is restored


def test_unwritable_cache_still_returns_built_algorithm(configured, monkeypatch):
    cache, _, _ = configured

    def denied(*args):
        raise PermissionError("read-only")

    monkeypatch.setattr(cache, "_write_atomic", denied)
    assert invoke(cache, lambda **kwargs: 42) == 42


def test_build_errors_are_not_hidden(configured):
    cache, _, _ = configured

    def fail(**kwargs):
        raise RuntimeError("build failed")

    with pytest.raises(RuntimeError, match="build failed"):
        invoke(cache, fail)


def test_missing_serialization_still_builds(cache, monkeypatch, tmp_path):
    monkeypatch.setenv("AWKWARD_CUDA_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache, "_serialization_api", lambda: (None, None))
    assert invoke(cache, lambda **kwargs: 42) == 42
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_cleans_up_after_replace_failure(cache, monkeypatch, tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"old")

    def denied(*args):
        raise OSError("replace failed")

    monkeypatch.setattr(cache.os, "replace", denied)
    with pytest.raises(OSError, match="replace failed"):
        cache._write_atomic(path, b"new")
    assert path.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [path]


def test_concurrent_atomic_writers_leave_complete_blob(cache, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "artifact.bin"
    payloads = [bytes([i]) * 10000 for i in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda p: cache._write_atomic(path, p), payloads))
    assert path.read_bytes() in payloads
    assert list(tmp_path.iterdir()) == [path]


def test_key_invalidates_code_dtype_architecture_runtime_and_versions(cache, tmp_path):
    source = tmp_path / "source.py"
    source.write_text("callback v1")
    args = [source, "<i8", "<i8", [8, 0], 12090, {"cccl": "1"}]
    original = cache._cache_key(*args)
    for position, value in [
        (1, "<i4"),
        (2, "<i4"),
        (3, [9, 0]),
        (4, 13000),
        (5, {"cccl": "2"}),
    ]:
        changed = args.copy()
        changed[position] = value
        assert cache._cache_key(*changed) != original
    source.write_text("callback v2")
    assert cache._cache_key(*args) != original


def test_unknown_package_version_is_absent(cache, monkeypatch):
    # Package discovery is covered without requiring CUDA on the test host.
    assert cache._version("this-package-does-not-exist-awkward-test") is None


def test_noncontiguous_arrays_bypass_persistent_cache(cache):
    from types import SimpleNamespace

    source = SimpleNamespace(ndim=1, flags=SimpleNamespace(c_contiguous=False))
    output = SimpleNamespace(ndim=1, flags=SimpleNamespace(c_contiguous=True))
    assert cache._cache_path("unused", __file__, source, output) is None


def test_serialization_failure_preserves_successful_build(configured, monkeypatch):
    cache, path, _ = configured

    def unsupported(algorithm):
        raise RuntimeError("serialization unavailable for this build")

    monkeypatch.setattr(
        cache, "_serialization_api", lambda: (unsupported, lambda b: None)
    )
    assert invoke(cache, lambda **kwargs: 42) == 42
    assert not path.exists()


def test_unidentified_cccl_version_disables_artifact(cache, monkeypatch):
    from types import SimpleNamespace

    from awkward._nplikes.cupy import Cupy

    monkeypatch.setattr(Cupy, "instance", lambda: SimpleNamespace(_module=None))
    monkeypatch.setattr(cache, "_version", lambda name: None)
    array = SimpleNamespace(ndim=1, flags=SimpleNamespace(c_contiguous=True))
    assert cache._cache_path("unused", __file__, array, array) is None


def test_api_discovery_failure_falls_back_to_factory(cache, monkeypatch, tmp_path):
    monkeypatch.setenv("AWKWARD_CUDA_CACHE_DIR", str(tmp_path))

    def unavailable():
        raise ImportError("serialization unavailable")

    monkeypatch.setattr(cache, "_serialization_api", unavailable)
    assert invoke(cache, lambda **kwargs: 42) == 42


def test_cache_path_separates_device_capability(cache, monkeypatch, tmp_path):
    from types import SimpleNamespace

    import numpy as np

    from awkward._nplikes.cupy import Cupy

    capability = {"major": 8, "minor": 0}
    runtime = SimpleNamespace(
        getDevice=lambda: 0,
        getDeviceProperties=lambda device: capability,
        runtimeGetVersion=lambda: 12090,
    )
    monkeypatch.setattr(
        Cupy,
        "instance",
        lambda: SimpleNamespace(
            _module=SimpleNamespace(cuda=SimpleNamespace(runtime=runtime))
        ),
    )
    monkeypatch.setattr(cache, "_version", lambda name: "1.3.0")
    array = np.arange(3, dtype=np.int64)
    first = cache._cache_path(tmp_path, __file__, array, array)
    assert first.parent == tmp_path / "indexed-fill-v1"
    assert first.suffix == ".bin"
    assert first == cache._cache_path(tmp_path, __file__, array.copy(), array.copy())
    capability["major"] = 9
    assert first != cache._cache_path(tmp_path, __file__, array, array)


@pytest.mark.parametrize("top_level", [True, False])
def test_serialization_api_locations(cache, monkeypatch, top_level):
    import sys
    from types import ModuleType

    parent = ModuleType("cuda")
    parent.__path__ = []
    compute = ModuleType("cuda.compute")
    compute.__path__ = []
    algorithms = ModuleType("cuda.compute.algorithms")
    serialize, deserialize = object(), object()
    target = compute if top_level else algorithms
    target.serialize, target.deserialize = serialize, deserialize
    parent.compute = compute
    compute.algorithms = algorithms
    for name, module in (
        ("cuda", parent),
        ("cuda.compute", compute),
        ("cuda.compute.algorithms", algorithms),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    assert cache._serialization_api() == (serialize, deserialize)
