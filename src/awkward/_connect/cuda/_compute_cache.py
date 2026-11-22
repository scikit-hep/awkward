# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Optional, disposable serialized artifacts for the IndexedArray fill transform."""

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from threading import Lock

# Windows replacement can fail when another thread is replacing the same file.
# Serialize only the short disk-write section, never compilation or launches.
_write_lock = Lock()


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cache_key(source_path, input_dtype, output_dtype, capability, runtime, versions):
    description = {
        "schema": 1,
        "algorithm": "IndexedArray_fill/unary_transform",
        "source": hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
        "cache_source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input": input_dtype,
        "output": output_dtype,
        "arrays": "contiguous/rank1",
        "state": "int64/rank1",
        "capability": capability,
        "runtime": runtime,
        "versions": versions,
        "python": list(sys.version_info[:3]),
        "platform": [sys.platform, platform.machine()],
    }
    return hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()


def _cache_path(directory, source_path, source, output):
    from awkward._nplikes.cupy import Cupy

    # This cache schema covers only the plain, contiguous rank-one array path.
    if any(
        array.ndim != 1 or not array.flags.c_contiguous for array in (source, output)
    ):
        return None
    cp = Cupy.instance()._module
    versions = {
        name: _version(name)
        for name in (
            "cuda-cccl",
            "cccl-python",
            "numba-cuda-mlir",
            "numba-cuda",
            "cupy",
            "cupy-cuda12x",
            "cupy-cuda13x",
        )
    }
    if versions["cuda-cccl"] is None and versions["cccl-python"] is None:
        return None
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    key = _cache_key(
        source_path,
        source.dtype.str,
        output.dtype.str,
        [properties["major"], properties["minor"]],
        cp.cuda.runtime.runtimeGetVersion(),
        versions,
    )
    return Path(directory).expanduser() / "indexed-fill-v1" / (key + ".bin")


def _serialization_api():
    import cuda.compute as cc

    if hasattr(cc, "serialize") and hasattr(cc, "deserialize"):
        return cc.serialize, cc.deserialize
    from cuda.compute import algorithms

    return getattr(algorithms, "serialize", None), getattr(
        algorithms, "deserialize", None
    )


def _write_atomic(path, payload):
    with _write_lock:
        _write_atomic_unlocked(path, payload)


def _write_atomic_unlocked(path, payload):
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_or_build(factory, *, source_path, d_in, d_out, op):
    """Reuse code only; the caller must supply its live operator on every launch."""
    directory = os.environ.get("AWKWARD_CUDA_CACHE_DIR")
    path = None
    serialize = deserialize = None
    if directory:
        try:
            serialize, deserialize = _serialization_api()
            if serialize is not None and deserialize is not None:
                path = _cache_path(directory, source_path, d_in, d_out)
                if path is not None and path.is_file():
                    return deserialize(path.read_bytes())
            else:
                serialize = None
        except Exception:
            # Artifacts are optional: missing, incompatible, or corrupt entries
            # must not prevent normal compilation. Never suppress build/launch errors.
            path = None

    algorithm = factory(d_in=d_in, d_out=d_out, op=op)
    if directory and serialize is not None:
        try:
            # Resolve again after a rejected artifact so a good build repairs it.
            if path is None:
                path = _cache_path(directory, source_path, d_in, d_out)
            if path is not None:
                _write_atomic(path, serialize(algorithm))
        except Exception:
            # Read-only/full filesystems or unsupported serializers disable reuse.
            pass
    return algorithm
