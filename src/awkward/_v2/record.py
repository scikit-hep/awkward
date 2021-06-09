# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import


class Record(dict):
    # FIX ME the documentation points to ak.layout.Record.__init__(array, at), but the test is sending a zip object
    def __init__(self, input):
        assert isinstance(input, zip)
        self.input = input

    def __getitem__(self, where):
        return dict(self.input)[where]
