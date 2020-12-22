# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


class Verbose(dict):
    def __getitem__(self, key):
        # print(key)
        self.touched = True
        if "data" in key:
            self.data_touched = True
        return dict.__getitem__(self, key)


def test_lazy():
    form = {
        "class": "RecordArray",
        "contents": {
            "Electron": {
                "class": "ListOffsetArray64",
                "content": {
                    "class": "RecordArray",
                    "contents": {
                        "charge": {
                            "class": "NumpyArray",
                            "form_key": "load_electron_charge",
                            "format": "i",
                            "has_identities": False,
                            "inner_shape": [],
                            "itemsize": 4,
                            "parameters": {},
                            "primitive": "int64",
                        },
                        "momentum": {
                            "class": "NumpyArray",
                            "form_key": "load_electron_momentum",
                            "format": "i",
                            "has_identities": False,
                            "inner_shape": [],
                            "itemsize": 4,
                            "parameters": {},
                            "primitive": "float64",
                        },
                    },
                    "form_key": "invalid",
                    "parameters": {},
                },
                "form_key": "load_electron_offsets",
                "offsets": "i64",
            },
            "Muon": {
                "class": "ListOffsetArray64",
                "content": {
                    "class": "RecordArray",
                    "contents": {
                        "charge": {
                            "class": "NumpyArray",
                            "form_key": "load_muon_charge",
                            "format": "i",
                            "has_identities": False,
                            "inner_shape": [],
                            "itemsize": 4,
                            "parameters": {},
                            "primitive": "int64",
                        },
                        "momentum": {
                            "class": "NumpyArray",
                            "form_key": "load_muon_momentum",
                            "format": "i",
                            "has_identities": False,
                            "inner_shape": [],
                            "itemsize": 4,
                            "parameters": {},
                            "primitive": "float64",
                        },
                    },
                    "form_key": "invalid",
                    "parameters": {"__record__": "Muon"},
                },
                "form_key": "load_muon_offsets",
                "offsets": "i64",
            },
        },
        "form_key": "",
        "parameters": {},
    }

    container = Verbose(
        {
            "part0-load_electron_offsets-offsets": np.array([0, 3, 3, 5]),
            "part0-load_muon_offsets-offsets": np.array([0, 3, 3, 5]),
            "part0-load_electron_charge-data": np.array(
                [1, 2, 3, 4, 5], dtype=np.int64
            ),
            "part0-load_electron_momentum-data": np.array(
                [1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64
            ),
            "part0-load_muon_charge-data": np.array([1, 2, 3, 4, 5], dtype=np.int64),
            "part0-load_muon_momentum-data": np.array(
                [1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64
            ),
            "part1-load_electron_offsets-offsets": np.array([0, 3, 3, 5]),
            "part1-load_muon_offsets-offsets": np.array([0, 3, 3, 5]),
            "part1-load_electron_charge-data": np.array(
                [1, 2, 3, 4, 5], dtype=np.int64
            ),
            "part1-load_electron_momentum-data": np.array(
                [1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64
            ),
            "part1-load_muon_charge-data": np.array([1, 2, 3, 4, 5], dtype=np.int64),
            "part1-load_muon_momentum-data": np.array(
                [1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64
            ),
        }
    )
    container.touched = False

    lazy = ak.from_buffers(form, [3, 3], container, lazy=True)
    one = ak.concatenate([lazy.Electron, lazy.Muon], axis=0)

    lazy = ak.from_buffers(form, [3, 3], container, lazy=True)
    two = ak.concatenate([lazy.Muon, lazy.Muon], axis=0)

    lazy = ak.from_buffers(form, [3, 3], container, lazy=True)
    three = ak.concatenate([lazy.Electron, lazy.Electron], axis=0)

    assert not container.touched

    assert one.tolist() == 4 * [
        [
            {"charge": 1, "momentum": 1.1},
            {"charge": 2, "momentum": 2.2},
            {"charge": 3, "momentum": 3.3},
        ],
        [],
        [{"charge": 4, "momentum": 4.4}, {"charge": 5, "momentum": 5.5}],
    ]
    assert two.tolist() == 4 * [
        [
            {"charge": 1, "momentum": 1.1},
            {"charge": 2, "momentum": 2.2},
            {"charge": 3, "momentum": 3.3},
        ],
        [],
        [{"charge": 4, "momentum": 4.4}, {"charge": 5, "momentum": 5.5}],
    ]
    assert three.tolist() == 4 * [
        [
            {"charge": 1, "momentum": 1.1},
            {"charge": 2, "momentum": 2.2},
            {"charge": 3, "momentum": 3.3},
        ],
        [],
        [{"charge": 4, "momentum": 4.4}, {"charge": 5, "momentum": 5.5}],
    ]

    assert container.touched

    container.data_touched = False

    lazy = ak.from_buffers(form, [3, 3], container, lazy=True)
    one = ak.concatenate([lazy.Electron, lazy.Muon], axis=1)

    lazy = ak.from_buffers(form, [3, 3], container, lazy=True)
    two = ak.concatenate([lazy.Muon, lazy.Muon], axis=1)

    lazy = ak.from_buffers(form, [3, 3], container, lazy=True)
    three = ak.concatenate([lazy.Electron, lazy.Electron], axis=1)

    assert not container.data_touched

    assert one.tolist() == 2 * [
        [
            {"charge": 1, "momentum": 1.1},
            {"charge": 2, "momentum": 2.2},
            {"charge": 3, "momentum": 3.3},
            {"charge": 1, "momentum": 1.1},
            {"charge": 2, "momentum": 2.2},
            {"charge": 3, "momentum": 3.3},
        ],
        [],
        [
            {"charge": 4, "momentum": 4.4},
            {"charge": 5, "momentum": 5.5},
            {"charge": 4, "momentum": 4.4},
            {"charge": 5, "momentum": 5.5},
        ]
    ]
    assert two.tolist() == 2 * [
        [
            {"charge": 1, "momentum": 1.1},
            {"charge": 2, "momentum": 2.2},
            {"charge": 3, "momentum": 3.3},
            {"charge": 1, "momentum": 1.1},
            {"charge": 2, "momentum": 2.2},
            {"charge": 3, "momentum": 3.3},
        ],
        [],
        [
            {"charge": 4, "momentum": 4.4},
            {"charge": 5, "momentum": 5.5},
            {"charge": 4, "momentum": 4.4},
            {"charge": 5, "momentum": 5.5},
        ]
    ]
    assert three.tolist() == 2 * [
        [
            {"charge": 1, "momentum": 1.1},
            {"charge": 2, "momentum": 2.2},
            {"charge": 3, "momentum": 3.3},
            {"charge": 1, "momentum": 1.1},
            {"charge": 2, "momentum": 2.2},
            {"charge": 3, "momentum": 3.3},
        ],
        [],
        [
            {"charge": 4, "momentum": 4.4},
            {"charge": 5, "momentum": 5.5},
            {"charge": 4, "momentum": 4.4},
            {"charge": 5, "momentum": 5.5},
        ]
    ]
