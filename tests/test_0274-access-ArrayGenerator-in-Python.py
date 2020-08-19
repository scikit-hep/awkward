# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import pytest
import numpy

import awkward1

def test():
    content = awkward1.Array([1, 2, 3, 4, 5])
    generator1 = awkward1.layout.ArrayGenerator(lambda a, b: a*content + b, (100,), {"b": 3})
    assert generator1.form is None
    assert awkward1.to_list(generator1()) == [103, 203, 303, 403, 503]
    assert generator1.args == (100,)
    assert generator1.kwargs == {"b": 3}
    assert generator1.form is not None
    assert generator1.length is None

    generator2 = generator1.with_args((1000,))
    assert generator2.form is None
    assert awkward1.to_list(generator2()) == [1003, 2003, 3003, 4003, 5003]
    assert generator2.args == (1000,)
    assert generator2.kwargs == {"b": 3}
    assert generator2.form is not None
    assert generator2.length is None

    generator3 = generator1.with_kwargs({"b": 4})
    assert generator3.form is None
    assert awkward1.to_list(generator3()) == [104, 204, 304, 404, 504]
    assert generator3.args == (100,)
    assert generator3.kwargs == {"b": 4}
    assert generator3.form is not None
    assert generator3.length is None

    generator4 = generator1.with_form(awkward1.forms.Form.fromjson('"int64"'))
    assert generator4.form is not None
    assert awkward1.to_list(generator4()) == [103, 203, 303, 403, 503]
    assert generator4.args == (100,)
    assert generator4.kwargs == {"b": 3}
    assert json.loads(generator4.form.tojson()) == json.loads(awkward1.forms.Form.fromjson('"int64"').tojson())
    assert generator4.length is None

    generator5 = generator1.with_length(5)
    assert generator5.form is None
    assert awkward1.to_list(generator5()) == [103, 203, 303, 403, 503]
    assert generator5.args == (100,)
    assert generator5.kwargs == {"b": 3}
    assert generator5.form is not None
    assert generator5.length == 5
