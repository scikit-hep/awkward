# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pickle

import pytest  # noqa: F401

from awkward._nplikes.shape import unknown_length


def test():
    assert pickle.loads(pickle.dumps(unknown_length)) is unknown_length
