# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import pytest
import numpy

import awkward1

def test_forms():
    form = awkward1.forms.BitMaskedForm("i8", awkward1.forms.NumpyForm([], 8, "d"), True, False, parameters={"hey": ["you"]})
    assert json.loads(form.tojson(False, True)) == {"mask": "i8", "content": {"inner_shape": [], "itemsize": 8, "format": "d", "dtype": "float64", "has_identities": False, "parameters": {}}, "valid_when": True, "lsb_order": False, "has_identities": False, "parameters": {"hey": ["you"]}}
