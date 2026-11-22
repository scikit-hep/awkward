# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from types import SimpleNamespace

import pytest

import awkward._connect.cuda as cuda


class Status:
    def __init__(self, value):
        self.value = value
        self.device = SimpleNamespace(id=0)
        self.reads = 0

    def get(self, *, stream=None):
        self.reads += 1
        return self

    def tolist(self):
        return self.value


@pytest.fixture
def state(monkeypatch):
    stream = SimpleNamespace(ptr=17, synchronize=lambda: None)
    monkeypatch.setattr(cuda, "cuda_streamptr_to_contexts", {})
    monkeypatch.setattr(
        cuda,
        "cupy",
        SimpleNamespace(
            array=Status, cuda=SimpleNamespace(get_current_stream=lambda: stream)
        ),
    )
    return stream


def decode(value):
    row, attempt = value
    return None if row == -1 else ValueError(f"index out of range at {row}: {attempt}")


def enqueue(stream, value, context=None):
    invocation = cuda.Invocation("test", context)
    status = Status(value)
    token = cuda._active_invocation.set((invocation, stream.ptr, 0))
    try:
        cuda.defer_status(status, decode, stream=stream, keepalive=(object(),))
    finally:
        cuda._active_invocation.reset(token)
    return invocation, status


def test_deferred_read_and_successful_drain(state):
    invocation, status = enqueue(state, [-1, 0])
    cuda.cuda_streamptr_to_contexts[state.ptr] = (Status(cuda.NO_ERROR), [invocation])
    assert status.reads == 0
    cuda.synchronize_cuda(state)
    assert status.reads == 1
    assert cuda.cuda_streamptr_to_contexts[state.ptr][1] == []


@pytest.mark.parametrize("legacy_first", [False, True])
def test_earliest_invocation_wins(state, monkeypatch, legacy_first):
    invocation, status = enqueue(state, [3, 99])
    legacy = cuda.Invocation("legacy", None)
    monkeypatch.setitem(cuda.kernel_errors, "legacy", ["legacy error"])
    contexts = [legacy, invocation] if legacy_first else [invocation, legacy]
    word = 0 if legacy_first else 1 << cuda.ERROR_BITS
    cuda.cuda_streamptr_to_contexts[state.ptr] = (Status(word), contexts)
    with pytest.raises(
        ValueError, match="legacy error" if legacy_first else "at 3: 99"
    ):
        cuda.synchronize_cuda(state)
    assert status.reads == (0 if legacy_first else 1)
    assert cuda.cuda_streamptr_to_contexts[state.ptr][1] == []


def test_context_and_decoder_failure_cleanup(state):
    class Context:
        def decorate_exception(self, cls, error):
            return cls(f"context: {error}")

    invocation, _ = enqueue(state, [2, -9], Context())
    cuda.cuda_streamptr_to_contexts[state.ptr] = (Status(cuda.NO_ERROR), [invocation])
    with pytest.raises(ValueError, match="context: index out of range at 2: -9"):
        cuda.synchronize_cuda(state)
    invocation.deferred_statuses[0] = (Status([]), decode, ())
    cuda.cuda_streamptr_to_contexts[state.ptr] = (Status(cuda.NO_ERROR), [invocation])
    with pytest.raises(ValueError):
        cuda.synchronize_cuda(state)
    assert cuda.cuda_streamptr_to_contexts[state.ptr][1] == []


def test_registration_requires_owner(state):
    with pytest.raises(RuntimeError, match="active"):
        cuda.defer_status(Status([-1, 0]), decode, stream=state)
    invocation = cuda.Invocation("test", None)
    token = cuda._active_invocation.set((invocation, 999, 0))
    try:
        with pytest.raises(ValueError, match="stream/device"):
            cuda.defer_status(Status([-1, 0]), decode, stream=state)
    finally:
        cuda._active_invocation.reset(token)
    assert invocation.deferred_statuses == []


@pytest.mark.parametrize("raises", [False, True])
def test_dispatch_scope_restored(state, monkeypatch, raises):
    from awkward._kernels import BaseKernel, CudaComputeKernel

    monkeypatch.setattr(
        cuda.cupy.cuda, "Device", lambda: SimpleNamespace(id=0), raising=False
    )
    status = Status([-1, 0])

    def impl():
        cuda.defer_status(status, decode, stream=state)
        if raises:
            raise RuntimeError("launch failed")

    kernel = object.__new__(CudaComputeKernel)
    BaseKernel.__init__(kernel, impl, ("test",))
    if raises:
        with pytest.raises(RuntimeError, match="launch failed"):
            kernel()
    else:
        kernel()
    assert cuda._active_invocation.get() is None
    assert status.reads == 0
    cuda.synchronize_cuda(state)
    assert status.reads == 1


def test_stream_isolation_and_scratch_lifetime(state):
    import weakref

    class Scratch:
        pass

    other = SimpleNamespace(ptr=18, synchronize=lambda: None)
    first, _ = enqueue(state, [-1, 0])
    second, _ = enqueue(other, [4, 10])
    scratch = Scratch()
    reference = weakref.ref(scratch)
    status, decoder, _ = first.deferred_statuses[0]
    first.deferred_statuses[0] = (status, decoder, (scratch,))
    cuda.cuda_streamptr_to_contexts[state.ptr] = (Status(cuda.NO_ERROR), [first])
    cuda.cuda_streamptr_to_contexts[other.ptr] = (Status(cuda.NO_ERROR), [second])
    del scratch, first
    assert reference() is not None
    cuda.synchronize_cuda(state)
    assert reference() is None
    assert len(cuda.cuda_streamptr_to_contexts[other.ptr][1]) == 1
    with pytest.raises(ValueError, match="at 4: 10"):
        cuda.synchronize_cuda(other)
