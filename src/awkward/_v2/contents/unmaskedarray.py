# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

from awkward._v2.contents.content import Content


class UnmaskedArray(Content):
    def __init__(self, content):
        assert isinstance(content, Content)
        self._content = content

    @property
    def contents(self):
        return self._contents

    def __len__(self):
        return len(self._content)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<UnmaskedArray len="]
        out.append(repr(str(len(self))))
        out.append(">\n")
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent)
        out.append("</UnmaskedArray>")
        out.append(post)
        return "".join(out)

    def _getitem_at(self, where):
        return self._content[where]

    def _getitem_range(self, where):
        return UnmaskedArray(self._content[where])

    def _getitem_field(self, where):
        return UnmaskedArray(self._content[where])

    def _getitem_fields(self, where):
        return UnmaskedArray(self._content[where])
