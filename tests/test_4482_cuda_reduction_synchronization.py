# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import ast
from pathlib import Path

import pytest


class DeviceOffsets:
    def __getitem__(self, index):
        raise AssertionError("offsets must reach CCCL without host inspection")


@pytest.mark.parametrize("segments", [0, 1, 100])
@pytest.mark.parametrize("hint", ["omitted", None, 0, 32, 1024])
def test_reductions_do_not_inspect_device_offsets(segments, hint):
    source = (
        Path(__file__).resolve().parents[1] / "src/awkward/_connect/cuda/_compute.py"
    )
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "segmented_reduce"
    )
    calls = []
    result = object()

    def reduce(**kwargs):
        calls.append(kwargs)
        return result

    namespace = {"_segmented_reduce_raw": reduce}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    start, stop = DeviceOffsets(), DeviceOffsets()
    options = {
        "num_segments": segments,
        "start_offsets_in": start,
        "end_offsets_in": stop,
    }
    if hint != "omitted":
        options["max_segment_size"] = hint
    for _ in range(5):
        assert namespace["segmented_reduce"](**options) is result
    assert len(calls) == 5
    for call in calls:
        assert call["start_offsets_in"] is start
        assert call["end_offsets_in"] is stop
        assert call["num_segments"] == segments
        assert call["max_segment_size"] == (None if hint == "omitted" else hint)
