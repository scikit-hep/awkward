# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import os
import json

import pytest
import numpy

import awkward1

def test_fromfile():
    array = awkward1.from_json('tests/samples/test-record-array.json')
    assert awkward1.to_list(array) == [
        [{'x': 1.1, 'y': []}],
        [{'x': 2.2, 'y': [1]}],
        [{'x': 3.3, 'y': [1, 2]}],
        [{'x': 4.4, 'y': [1, 2, 3]}],
        [{'x': 5.5, 'y': [1, 2, 3, 4]}],
        [{'x': 6.6, 'y': [1, 2, 3, 4, 5]}]]

    array = awkward1.from_json('tests/samples/test.json')

    assert awkward1.to_list(array) == [1.1, 2.2, 3.3, 99999.99999, -99999.99999,
        [4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11],
        [12.12, 13.13, 14.14, 15.15, 16.16, 17.17],
        [[[18.18,   19.19,   20.2,    21.21,   22.22],
          [23.23,   24.24,   25.25,   26.26,   27.27,
           28.28,   29.29,   30.3,    31.31,   32.32,
           33.33,   34.34,   35.35,   36.36,   37.37],
          [38.38],
          [39.39, 40.4, 99999.99999, 99999.99999, 41.41, 42.42, 43.43]],
         [[44.44,   45.45,   46.46,   47.47,   48.48],
          [49.49,   50.5,    51.51,   52.52,   53.53,
           54.54,   55.55,   56.56,   57.57,   58.58,
           59.59,   60.6,    61.61,   62.62,   63.63],
          [64.64],
          [65.65, 66.66, 99999.99999, 99999.99999, 67.67, 68.68, 69.69]]]]
