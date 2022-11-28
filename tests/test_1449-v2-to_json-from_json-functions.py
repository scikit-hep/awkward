# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import base64
import os

import numpy as np  # noqa: F401
import pytest

import awkward as ak


def test_without_control():
    array = ak.Array(
        [
            {"ok": 1, "x": 1.1, "y": 1 + 1j, "z": b"one"},
            {"ok": 2, "x": 2.2, "y": 2 + 2j, "z": b"two"},
            {"ok": 3, "x": 3.3, "y": 3 + 3j, "z": b"three"},
            {"ok": 4, "x": float("nan"), "y": float("nan"), "z": b"four"},
            {"ok": 5, "x": float("inf"), "y": float("inf") + 5j, "z": b"five"},
            {"ok": 6, "x": float("-inf"), "y": 6 + float("-inf") * 1j, "z": b"six"},
            {"ok": 7, "x": 7.7, "y": 7 + 7j, "z": b"seven"},
            {"ok": 8, "x": None, "y": 8 + 8j, "z": b"eight"},
            {"ok": 9, "x": 9.9, "y": 9 + 9j, "z": b"nine"},
        ]
    )

    assert ak.to_json(array.ok) == "[1,2,3,4,5,6,7,8,9]"

    with pytest.raises(ValueError):
        ak.to_json(array.x)

    assert ak.to_json(array.x[:3]) == "[1.1,2.2,3.3]"

    with pytest.raises(ValueError):
        ak.to_json(array.x, nan_string="NAN")

    with pytest.raises(ValueError):
        ak.to_json(array.x, nan_string="NAN", posinf_string="INF")

    assert (
        ak.to_json(array.x, nan_string="NAN", posinf_string="INF", neginf_string="-INF")
        == '[1.1,2.2,3.3,"NAN","INF","-INF",7.7,null,9.9]'
    )

    with pytest.raises(TypeError):
        ak.to_json(array.y[:3])

    assert (
        ak.to_json(array.y[:3], complex_record_fields=["R", "I"])
        == '[{"R":1.0,"I":1.0},{"R":2.0,"I":2.0},{"R":3.0,"I":3.0}]'
    )

    with pytest.raises(TypeError):
        ak.to_json(array.z)

    assert (
        ak.to_json(array.z, convert_bytes=lambda x: x.decode())
        == '["one","two","three","four","five","six","seven","eight","nine"]'
    )


def test_to_json_options(tmp_path):
    filename = os.path.join(tmp_path, "whatever.json")

    array = ak.Array(
        [
            {"x": 1.1, "y": 1 + 1j, "z": b"one"},
            {"x": 2.2, "y": 2 + 2j, "z": b"two"},
            {"x": 3.3, "y": 3 + 3j, "z": b"three"},
            {"x": float("nan"), "y": float("nan"), "z": b"four"},
            {"x": float("inf"), "y": float("inf") + 5j, "z": b"five"},
            {"x": float("-inf"), "y": 6 + float("-inf") * 1j, "z": b"six"},
            {"x": 7.7, "y": 7 + 7j, "z": b"seven"},
            {"x": None, "y": 8 + 8j, "z": b"eight"},
            {"x": 9.9, "y": 9 + 9j, "z": b"nine"},
        ]
    )

    kwargs = {
        "nan_string": "nan",
        "posinf_string": "inf",
        "neginf_string": "-inf",
        "complex_record_fields": ("real", "imag"),
        "convert_bytes": lambda x: base64.b64encode(x).decode(),
    }

    expectation = '[{"x":1.1,"y":{"real":1.0,"imag":1.0},"z":"b25l"},{"x":2.2,"y":{"real":2.0,"imag":2.0},"z":"dHdv"},{"x":3.3,"y":{"real":3.0,"imag":3.0},"z":"dGhyZWU="},{"x":"nan","y":{"real":"nan","imag":0.0},"z":"Zm91cg=="},{"x":"inf","y":{"real":"inf","imag":5.0},"z":"Zml2ZQ=="},{"x":"-inf","y":{"real":"nan","imag":"-inf"},"z":"c2l4"},{"x":7.7,"y":{"real":7.0,"imag":7.0},"z":"c2V2ZW4="},{"x":null,"y":{"real":8.0,"imag":8.0},"z":"ZWlnaHQ="},{"x":9.9,"y":{"real":9.0,"imag":9.0},"z":"bmluZQ=="}]'

    assert ak.to_json(array, **kwargs) == expectation

    ak.to_json(array, filename, **kwargs)
    with open(filename) as file:
        assert file.read() == expectation

    with open(filename, "w") as file:
        ak.to_json(array, file, **kwargs)
    with open(filename) as file:
        assert file.read() == expectation

    expectation = '{"x":1.1,"y":{"real":1.0,"imag":1.0},"z":"b25l"}'

    assert ak.to_json(array[0], **kwargs) == expectation

    ak.to_json(array[0], filename, **kwargs)
    with open(filename) as file:
        assert file.read() == expectation

    with open(filename, "w") as file:
        ak.to_json(array[0], file, **kwargs)
    with open(filename) as file:
        assert file.read() == expectation

    expectation = """[
    {
        "x": 1.1,
        "y": {
            "real": 1.0,
            "imag": 1.0
        },
        "z": "b25l"
    },
    {
        "x": 2.2,
        "y": {
            "real": 2.0,
            "imag": 2.0
        },
        "z": "dHdv"
    },
    {
        "x": 3.3,
        "y": {
            "real": 3.0,
            "imag": 3.0
        },
        "z": "dGhyZWU="
    },
    {
        "x": "nan",
        "y": {
            "real": "nan",
            "imag": 0.0
        },
        "z": "Zm91cg=="
    },
    {
        "x": "inf",
        "y": {
            "real": "inf",
            "imag": 5.0
        },
        "z": "Zml2ZQ=="
    },
    {
        "x": "-inf",
        "y": {
            "real": "nan",
            "imag": "-inf"
        },
        "z": "c2l4"
    },
    {
        "x": 7.7,
        "y": {
            "real": 7.0,
            "imag": 7.0
        },
        "z": "c2V2ZW4="
    },
    {
        "x": null,
        "y": {
            "real": 8.0,
            "imag": 8.0
        },
        "z": "ZWlnaHQ="
    },
    {
        "x": 9.9,
        "y": {
            "real": 9.0,
            "imag": 9.0
        },
        "z": "bmluZQ=="
    }
]"""

    assert (
        ak.to_json(
            array, num_indent_spaces=4, num_readability_spaces=1, **kwargs
        ).replace(" \n", "\n")
        == expectation
    )

    ak.to_json(array, filename, num_indent_spaces=4, num_readability_spaces=1, **kwargs)
    with open(filename) as file:
        assert file.read().replace(" \n", "\n") == expectation

    with open(filename, "w") as file:
        ak.to_json(array, file, num_indent_spaces=4, num_readability_spaces=1, **kwargs)
    with open(filename) as file:
        assert file.read().replace(" \n", "\n") == expectation

    expectation = """{
    "x": 1.1,
    "y": {
        "real": 1.0,
        "imag": 1.0
    },
    "z": "b25l"
}"""

    assert (
        ak.to_json(
            array[0], num_indent_spaces=4, num_readability_spaces=1, **kwargs
        ).replace(" \n", "\n")
        == expectation
    )

    ak.to_json(
        array[0], filename, num_indent_spaces=4, num_readability_spaces=1, **kwargs
    )
    with open(filename) as file:
        assert file.read().replace(" \n", "\n") == expectation

    with open(filename, "w") as file:
        ak.to_json(
            array[0], file, num_indent_spaces=4, num_readability_spaces=1, **kwargs
        )
    with open(filename) as file:
        assert file.read().replace(" \n", "\n") == expectation

    expectation = """{"x":1.1,"y":{"real":1.0,"imag":1.0},"z":"b25l"}
{"x":2.2,"y":{"real":2.0,"imag":2.0},"z":"dHdv"}
{"x":3.3,"y":{"real":3.0,"imag":3.0},"z":"dGhyZWU="}
{"x":"nan","y":{"real":"nan","imag":0.0},"z":"Zm91cg=="}
{"x":"inf","y":{"real":"inf","imag":5.0},"z":"Zml2ZQ=="}
{"x":"-inf","y":{"real":"nan","imag":"-inf"},"z":"c2l4"}
{"x":7.7,"y":{"real":7.0,"imag":7.0},"z":"c2V2ZW4="}
{"x":null,"y":{"real":8.0,"imag":8.0},"z":"ZWlnaHQ="}
{"x":9.9,"y":{"real":9.0,"imag":9.0},"z":"bmluZQ=="}
"""

    assert ak.to_json(array, line_delimited=True, **kwargs) == expectation

    ak.to_json(array, filename, line_delimited=True, **kwargs)
    with open(filename) as file:
        assert file.read() == expectation

    with open(filename, "w") as file:
        ak.to_json(array, file, line_delimited=True, **kwargs)
    with open(filename) as file:
        assert file.read() == expectation

    expectation = """{"x":1.1,"y":{"real":1.0,"imag":1.0},"z":"b25l"}
"""

    assert ak.to_json(array[0], line_delimited=True, **kwargs) == expectation

    ak.to_json(array[0], filename, line_delimited=True, **kwargs)
    with open(filename) as file:
        assert file.read() == expectation

    with open(filename, "w") as file:
        ak.to_json(array[0], file, line_delimited=True, **kwargs)
    with open(filename) as file:
        assert file.read() == expectation

    expectation = """{"x":1.1,"y":{"real":1.0,"imag":1.0},"z":"b25l"}
{"x":2.2,"y":{"real":2.0,"imag":2.0},"z":"dHdv"}
{"x":3.3,"y":{"real":3.0,"imag":3.0},"z":"dGhyZWU="}
{"x":"nan","y":{"real":"nan","imag":0.0},"z":"Zm91cg=="}
{"x":"inf","y":{"real":"inf","imag":5.0},"z":"Zml2ZQ=="}
{"x":"-inf","y":{"real":"nan","imag":"-inf"},"z":"c2l4"}
{"x":7.7,"y":{"real":7.0,"imag":7.0},"z":"c2V2ZW4="}
{"x":null,"y":{"real":8.0,"imag":8.0},"z":"ZWlnaHQ="}
{"x":9.9,"y":{"real":9.0,"imag":9.0},"z":"bmluZQ=="}
"""

    with open(filename, "w") as file:
        for x in array:
            ak.to_json(x, file, line_delimited=True, **kwargs)
    with open(filename) as file:
        assert file.read() == expectation
