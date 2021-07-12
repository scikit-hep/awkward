# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.forms.form import Form


class IndexedForm(Form):
    def __init__(
        self,
        index,
        content=None,
        has_identifier=False,
        parameters=None,
        form_key=None,
    ):
        if not ak._util.isstr(index):
            raise TypeError(
                "{0} 'index' must be of type str, not {1}".format(
                    type(self).__name__, repr(index)
                )
            )
        if not isinstance(content, Form):
            raise TypeError(
                "{0} all 'contents' must be Form subclasses, not {1}".format(
                    type(self).__name__, repr(content)
                )
            )

        self._index = index
        self._content = content
        self._init(has_identifier, parameters, form_key)

    @property
    def index(self):
        return self._index

    @property
    def content(self):
        return self._content

    def __repr__(self):
        args = [repr(self._index), repr(self._content)] + self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def _tolist_part(self, verbose, toplevel):
        return self._tolist_extra(
            {
                "class": "IndexedArray",
                "index": self._index,
                "content": self._content._tolist_part(verbose, toplevel=False),
            },
            verbose,
        )
