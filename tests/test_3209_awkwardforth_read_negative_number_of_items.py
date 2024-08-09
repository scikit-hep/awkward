# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak


def test_read_negative_number_of_items():
    vm = ak.forth.ForthMachine32("input source -5 source #q-> stack")
    vm.run({"source": np.array([1, 2, 3, 4, 5], dtype=np.int64)})
    assert vm.stack == []

    vm = ak.forth.ForthMachine32("input source output sink float64 -5 source #q-> sink")
    vm.run({"source": np.array([1, 2, 3, 4, 5], dtype=np.int64)})
    assert vm.output("sink").tolist() == []


def test_read_negative_and_positive_number_of_items():
    vm = ak.forth.ForthMachine32(
        "input source -5 source #q-> stack 5 source #q-> stack"
    )
    vm.run({"source": np.array([1, 2, 3, 4, 5], dtype=np.int64)})
    assert vm.stack == [1, 2, 3, 4, 5]

    vm = ak.forth.ForthMachine32(
        "input source output sink float64 -5 source #q-> sink 5 source #q-> sink"
    )
    vm.run({"source": np.array([1, 2, 3, 4, 5], dtype=np.int64)})
    assert vm.output("sink").tolist() == [1, 2, 3, 4, 5]


def test_read_positive_and_negative_number_of_items():
    vm = ak.forth.ForthMachine32(
        "input source 5 source #q-> stack -5 source #q-> stack"
    )
    vm.run({"source": np.array([1, 2, 3, 4, 5], dtype=np.int64)})
    assert vm.stack == [1, 2, 3, 4, 5]

    vm = ak.forth.ForthMachine32(
        "input source output sink float64 5 source #q-> sink -5 source #q-> sink"
    )
    vm.run({"source": np.array([1, 2, 3, 4, 5], dtype=np.int64)})
    assert vm.output("sink").tolist() == [1, 2, 3, 4, 5]
