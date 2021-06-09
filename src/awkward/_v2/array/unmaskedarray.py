# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

from awkward._v2.content import Content


class UnmaskedArray(Content):
    def __init__(self, content):
        assert isinstance(content, Content)
        self.content = content

    def __len__(self):
        return len(self.content)

    def __repr__(self):
        return "UnmaskedArray(" + repr(self.content) + ")"

    def _getitem_at(self, where):
        return self.content[where]

    def _getitem_range(self, where):
        return UnmaskedArray(self.content[where])

    def _getitem_field(self, where):
        return UnmaskedArray(self.content[where])

    def _getitem_fields(self, where):
        return UnmaskedArray(self.content[where])
