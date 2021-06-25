# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.forms.form import Form


class ListOffsetForm(Form):
    def __init__(
        self, offsets, content, has_identifier=False, parameters=None, form_key=None
    ):
        if not ak._util.isstr(offsets):
            raise TypeError(
                "{0} 'offsets' must be of type str, not {1}".format(
                    type(self).__name__, repr(offsets)
                )
            )

        self._offsets = offsets
        self._content = content
        self._init(has_identifier, parameters, form_key)

    @property
    def offsets(self):
        return self._offsets

    @property
    def content(self):
        return self._content

    def __repr__(self):
        args = [
            repr(self._offsets),
            repr(self._content),
        ] + self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def _tolist_part(self, verbose, toplevel):
        return self._tolist_extra(
            {
                "class": "ListOffsetArray",
                "offsets": self._offsets,
                "content": self._content._tolist_part(verbose, toplevel=False),
            },
            verbose,
        )
