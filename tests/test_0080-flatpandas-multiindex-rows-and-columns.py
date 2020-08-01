# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json
import distutils.version

import pytest
import numpy

import awkward1

pandas = pytest.importorskip("pandas")

@pytest.mark.skipif(distutils.version.LooseVersion(pandas.__version__) < distutils.version.LooseVersion("1.0"), reason="Test Pandas in 1.0+ because they had to fix their JSON format.")
def test():
    def key(n):
        if n == "values":
            return n
        else:
            return tuple(eval(n.replace("nan", "None").replace("null", "None")))

    def regularize(data):
        if isinstance(data, dict):
            return dict((key(n), regularize(x)) for n, x in data.items())
        else:
            return data

    array = awkward1.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, None, 8.8, 9.9]])
    assert regularize(json.loads(awkward1.to_pandas(array).to_json())) == {"values": {(0, 0): 0.0, (0, 1): 1.1, (0, 2): 2.2, (2, 0): 3.3, (2, 1): 4.4, (3, 0): 5.5, (4, 0): 6.6, (4, 1): None, (4, 2): 8.8, (4, 3): 9.9}}

    array = awkward1.Array([[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [[5.5]], [[6.6, None, 8.8, 9.9]]])
    assert regularize(json.loads(awkward1.to_pandas(array).to_json())) == {"values": {(0, 0, 0): 0.0, (0, 0, 1): 1.1, (0, 0, 2): 2.2, (0, 2, 0): 3.3, (0, 2, 1): 4.4, (1, 0, 0): 5.5, (2, 0, 0): 6.6, (2, 0, 1): None, (2, 0, 2): 8.8, (2, 0, 3): 9.9}}

    array = awkward1.Array([[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5]], None, [[], None, [6.6, None, 8.8, 9.9]]])
    assert regularize(json.loads(awkward1.to_pandas(array).to_json())) == {"values": {(0, 0, 0): 0.0, (0, 0, 1): 1.1, (0, 0, 2): 2.2, (0, 2, 0): 3.3, (0, 2, 1): 4.4, (2, 0, 0): 5.5, (4, 2, 0): 6.6, (4, 2, 1): None, (4, 2, 2): 8.8, (4, 2, 3): 9.9}}

    array = awkward1.Array([[[{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}], [], [{"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}]], [], [[{"x": 5.5, "y": [5, 5, 5, 5, 5]}]]])
    assert regularize(json.loads(awkward1.to_pandas(array).to_json())) == {("x",): {(0, 0, 1, 0): 1.1, (0, 0, 2, 0): 2.2, (0, 0, 2, 1): 2.2, (0, 2, 0, 0): 3.3, (0, 2, 0, 1): 3.3, (0, 2, 0, 2): 3.3, (0, 2, 1, 0): 4.4, (0, 2, 1, 1): 4.4, (0, 2, 1, 2): 4.4, (0, 2, 1, 3): 4.4, (2, 0, 0, 0): 5.5, (2, 0, 0, 1): 5.5, (2, 0, 0, 2): 5.5, (2, 0, 0, 3): 5.5, (2, 0, 0, 4): 5.5}, ("y",): {(0, 0, 1, 0): 1, (0, 0, 2, 0): 2, (0, 0, 2, 1): 2, (0, 2, 0, 0): 3, (0, 2, 0, 1): 3, (0, 2, 0, 2): 3, (0, 2, 1, 0): 4, (0, 2, 1, 1): 4, (0, 2, 1, 2): 4, (0, 2, 1, 3): 4, (2, 0, 0, 0): 5, (2, 0, 0, 1): 5, (2, 0, 0, 2): 5, (2, 0, 0, 3): 5, (2, 0, 0, 4): 5}}

    assert regularize(json.loads(awkward1.to_pandas(array, how="outer").to_json())) == {("x",): {(0, 0, 0, None): 0.0, (0, 0, 1, 0.0): 1.1, (0, 0, 2, 0.0): 2.2, (0, 0, 2, 1.0): 2.2, (0, 2, 0, 0.0): 3.3, (0, 2, 0, 1.0): 3.3, (0, 2, 0, 2.0): 3.3, (0, 2, 1, 0.0): 4.4, (0, 2, 1, 1.0): 4.4, (0, 2, 1, 2.0): 4.4, (0, 2, 1, 3.0): 4.4, (2, 0, 0, 0.0): 5.5, (2, 0, 0, 1.0): 5.5, (2, 0, 0, 2.0): 5.5, (2, 0, 0, 3.0): 5.5, (2, 0, 0, 4.0): 5.5}, ("y",): {(0, 0, 0, None): None, (0, 0, 1, 0.0): 1.0, (0, 0, 2, 0.0): 2.0, (0, 0, 2, 1.0): 2.0, (0, 2, 0, 0.0): 3.0, (0, 2, 0, 1.0): 3.0, (0, 2, 0, 2.0): 3.0, (0, 2, 1, 0.0): 4.0, (0, 2, 1, 1.0): 4.0, (0, 2, 1, 2.0): 4.0, (0, 2, 1, 3.0): 4.0, (2, 0, 0, 0.0): 5.0, (2, 0, 0, 1.0): 5.0, (2, 0, 0, 2.0): 5.0, (2, 0, 0, 3.0): 5.0, (2, 0, 0, 4.0): 5.0}}

    array = awkward1.Array([[[{"x": 0.0, "y": 0}, {"x": 1.1, "y": 1}, {"x": 2.2, "y": 2}], [], [{"x": 3.3, "y": 3}, {"x": 4.4, "y": 4}]], [], [[{"x": 5.5, "y": 5}]]])
    assert regularize(json.loads(awkward1.to_pandas(array).to_json())) == {("x",): {(0, 0, 0): 0.0, (0, 0, 1): 1.1, (0, 0, 2): 2.2, (0, 2, 0): 3.3, (0, 2, 1): 4.4, (2, 0, 0): 5.5}, ("y",): {(0, 0, 0): 0, (0, 0, 1): 1, (0, 0, 2): 2, (0, 2, 0): 3, (0, 2, 1): 4, (2, 0, 0): 5}}

    array = awkward1.Array([[[{"x": 0.0, "y": {"z": 0}}, {"x": 1.1, "y": {"z": 1}}, {"x": 2.2, "y": {"z": 2}}], [], [{"x": 3.3, "y": {"z": 3}}, {"x": 4.4, "y": {"z": 4}}]], [], [[{"x": 5.5, "y": {"z": 5}}]]])
    assert regularize(json.loads(awkward1.to_pandas(array).to_json())) == {("x", ""): {(0, 0, 0): 0.0, (0, 0, 1): 1.1, (0, 0, 2): 2.2, (0, 2, 0): 3.3, (0, 2, 1): 4.4, (2, 0, 0): 5.5}, ("y", "z"): {(0, 0, 0): 0, (0, 0, 1): 1, (0, 0, 2): 2, (0, 2, 0): 3, (0, 2, 1): 4, (2, 0, 0): 5}}

    one = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    two = awkward1.Array([[100, 200], [300], [400, 500]])
    assert [regularize(json.loads(x.to_json())) for x in awkward1.to_pandas({"x": one, "y": two}, how=None)] == [
        {('x',): {(0, 0): 1.1, (0, 1): 2.2, (0, 2): 3.3, (2, 0):4.4, (2, 1): 5.5}},
        {('y',): {(0, 0): 100, (0, 1): 200, (1, 0): 300, (2, 0):400, (2, 1): 500}}]
