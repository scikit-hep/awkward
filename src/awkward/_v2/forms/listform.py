# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.forms.form import Form, parameters_equal


class ListForm(Form):
    def __init__(
        self,
        starts,
        stops,
        content,
        has_identifier=False,
        parameters=None,
        form_key=None,
    ):
        if not ak._util.isstr(starts):
            raise TypeError(
                "{0} 'starts' must be of type str, not {1}".format(
                    type(self).__name__, repr(starts)
                )
            )
        if not ak._util.isstr(stops):
            raise TypeError(
                "{0} 'starts' must be of type str, not {1}".format(
                    type(self).__name__, repr(starts)
                )
            )
        if not isinstance(content, Form):
            raise TypeError(
                "{0} all 'contents' must be Form subclasses, not {1}".format(
                    type(self).__name__, repr(content)
                )
            )

        self._starts = starts
        self._stops = stops
        self._content = content
        self._init(has_identifier, parameters, form_key)

    @property
    def starts(self):
        return self._starts

    @property
    def stops(self):
        return self._stops

    @property
    def content(self):
        return self._content

    def __repr__(self):
        args = [
            repr(self._starts),
            repr(self._stops),
            repr(self._content),
        ] + self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def _tolist_part(self, verbose, toplevel):
        return self._tolist_extra(
            {
                "class": "ListArray",
                "starts": self._starts,
                "stops": self._stops,
                "content": self._content._tolist_part(verbose, toplevel=False),
            },
            verbose,
        )

    def generated_compatibility(self, layout):
        from awkward._v2.contents.listarray import ListArray

        if isinstance(layout, ListArray):
            return (
                self._starts == layout.starts.form
                and self._stops == layout.stops.form
                and parameters_equal(self.parameters, layout.parameters)
                and self._content.generated_compatibility(layout.content)
            )
        else:
            return False

    @property
    def purelist_isregular(self):
        return False

    @property
    def purelist_depth(self):
        if self.parameter("__array__") in ("string", "bytestring"):
            return 1
        else:
            return self._content.purelist_depth + 1

    @property
    def minmax_depth(self):
        if self.parameter("__array__") in ("string", "bytestring"):
            return (1, 1)
        else:
            mindepth, maxdepth = self._content.minmax_depth
            return (mindepth + 1, maxdepth + 1)

    @property
    def branch_depth(self):
        if self.parameter("__array__") in ("string", "bytestring"):
            return (False, 1)
        else:
            branch, depth = self._content.branch_depth
            return (branch, depth + 1)

    @property
    def keys(self):
        return self._content.keys
