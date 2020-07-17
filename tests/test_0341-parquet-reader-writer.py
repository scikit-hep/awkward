# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_write_read(tmp_path):
    array1 = awkward1.Array([[1, 2, 3], [], [4, 5], [], [], [6, 7, 8, 9]])
    array2 = awkward1.repartition(array1, 2)
    array3 = awkward1.Array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3},
                             {"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}, {"x": 6, "y": 6.6},
                             {"x": 7, "y": 7.7}, {"x": 8, "y": 8.8}, {"x": 9, "y": 9.9}])
    array4 = awkward1.repartition(array3, 2)

    awkward1.to_parquet(array1, "array1.parquet")
    awkward1.to_parquet(array2, "array2.parquet")
    awkward1.to_parquet(array3, "array3.parquet")
    awkward1.to_parquet(array4, "array4.parquet")

    assert awkward1.to_list(awkward1.from_parquet("array1.parquet")) == awkward1.to_list(array1)
    assert awkward1.to_list(awkward1.from_parquet("array2.parquet")) == awkward1.to_list(array2)
    assert awkward1.to_list(awkward1.from_parquet("array3.parquet")) == awkward1.to_list(array3)
    assert awkward1.to_list(awkward1.from_parquet("array4.parquet")) == awkward1.to_list(array4)

    assert awkward1.to_list(awkward1.from_parquet("array1.parquet", lazy=True)) == awkward1.to_list(array1)
    assert awkward1.to_list(awkward1.from_parquet("array2.parquet", lazy=True)) == awkward1.to_list(array2)
    assert awkward1.to_list(awkward1.from_parquet("array3.parquet", lazy=True)) == awkward1.to_list(array3)
    assert awkward1.to_list(awkward1.from_parquet("array4.parquet", lazy=True)) == awkward1.to_list(array4)
