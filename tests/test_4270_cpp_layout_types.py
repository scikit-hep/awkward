# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import awkward as ak


@pytest.fixture(scope="module")
def exported_layouts(tmp_path_factory):
    if sys.platform == "emscripten":
        pytest.skip("Emscripten does not support subprocess execution")
    executable = os.environ.get("AWKWARD_EXTENDED_LAYOUT_TEST_EXECUTABLE")
    if executable is None:
        compiler = shutil.which("c++")
        if compiler is None:
            pytest.skip(
                "requires a C++23 compiler or a prebuilt layout test executable"
            )
        root = Path(__file__).resolve().parents[1]
        executable = str(tmp_path_factory.mktemp("extended-layout") / "export")
        subprocess.run(
            [
                compiler,
                "-std=c++23",
                *[
                    f"-I{root / 'header-only' / component}"
                    for component in (
                        "layout-builder",
                        "growable-buffer",
                        "builder-options",
                    )
                ],
                str(root / "header-only/tests/test_4270-extended-layout-types.cpp"),
                "-o",
                executable,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    result = subprocess.run([executable], check=True, capture_output=True, text=True)
    return [json.loads(line) for line in result.stdout.splitlines()]


@pytest.mark.parametrize("kind", ["float16", "longdouble_complex"])
def test_cpp_extended_layout_roundtrip(exported_layouts, kind):
    assert len(exported_layouts) in (2, 4)
    cases = []
    for case in exported_layouts:
        leaf = case["form"].get("content", case["form"])
        if (leaf["primitive"] == "float16") == (kind == "float16"):
            cases.append(case)
    if not cases:
        pytest.skip(f"compiler does not provide {kind}")
    if kind == "longdouble_complex":
        info = np.finfo(np.longdouble)
        for case in cases:
            if (
                case["component_size"] != np.dtype(np.longdouble).itemsize
                or case["component_mantissa"] != info.nmant + 1
                or case["component_max_exponent"] != info.maxexp
            ):
                pytest.skip(
                    "compiler long double representation differs from NumPy; "
                    "native raw buffers cannot be interpreted without conversion"
                )
    for case in cases:
        form = ak.forms.from_dict(case["form"])
        buffers = {key: bytes.fromhex(value) for key, value in case["buffers"].items()}
        array = ak.from_buffers(
            form, case["length"], buffers, byteorder="<" if np.little_endian else ">"
        )
        assert ak.validity_error(array) == ""
        assert array.layout.to_typetracer(forget_length=True).form == form
        if form.is_list:
            assert ak.num(array).to_list() == [
                6 if form.content.primitive == "float16" else 2,
                0,
            ]
            values = ak.to_numpy(ak.flatten(array))
        else:
            values = ak.to_numpy(array)
        if values.dtype.kind == "c":
            assert values.dtype == np.dtype(np.clongdouble)
            expected = np.empty(2, dtype=np.clongdouble)
            expected.real = [np.longdouble(1) + np.ldexp(np.longdouble(1), -60), -0.0]
            expected.imag = [-2.5, 3]
            np.testing.assert_array_equal(values, expected)
            assert np.signbit(values[1].real)
        else:
            assert values.dtype == np.dtype(np.float16)
            np.testing.assert_array_equal(
                values, [-0.0, 2**-24, 1.5, 65504, np.inf, np.nan]
            )
            assert np.signbit(values[0])
