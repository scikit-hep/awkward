# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward as ak
from awkward._v2.forms.form import Form, parameters_equal


class UnionForm(Form):
    def __init__(
        self,
        tags,
        index,
        contents,
        has_identifier=False,
        parameters=None,
        form_key=None,
    ):
        if not ak._util.isstr(tags):
            raise TypeError(
                "{0} 'tags' must be of type str, not {1}".format(
                    type(self).__name__, repr(tags)
                )
            )
        if not ak._util.isstr(index):
            raise TypeError(
                "{0} 'index' must be of type str, not {1}".format(
                    type(self).__name__, repr(index)
                )
            )
        if not isinstance(contents, Iterable):
            raise TypeError(
                "{0} 'contents' must be iterable, not {1}".format(
                    type(self).__name__, repr(contents)
                )
            )
        for content in contents:
            if not isinstance(content, Form):
                raise TypeError(
                    "{0} all 'contents' must be Form subclasses, not {1}".format(
                        type(self).__name__, repr(content)
                    )
                )

        self._tags = tags
        self._index = index
        self._contents = list(contents)
        self._init(has_identifier, parameters, form_key)

    @property
    def tags(self):
        return self._tags

    @property
    def index(self):
        return self._index

    @property
    def contents(self):
        return self._contents

    def content(self, index):
        return self._contents[index]

    def __repr__(self):
        args = [
            repr(self._tags),
            repr(self._index),
            repr(self._contents),
        ] + self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def _tolist_part(self, verbose, toplevel):
        return self._tolist_extra(
            {
                "class": "UnionArray",
                "tags": self._tags,
                "index": self._index,
                "contents": [
                    content._tolist_part(verbose, toplevel=False)
                    for content in self._contents
                ],
            },
            verbose,
        )

    def generated_compatibility(self, layout):
        from awkward._v2.contents.unionarray import UnionArray

        if isinstance(layout, UnionArray):
            if len(self._contents) == len(layout.contents):
                return parameters_equal(self._parameters, layout.parameters) and all(
                    x.generated_compatibility(y)
                    for x, y in zip(self._contents, layout.contents)
                )
            else:
                return False
        else:
            return False

    @property
    def purelist_isregular(self):
        for content in self._contents:
            if not content.purelist_isregular:
                return False
        else:
            return True

    @property
    def purelist_depth(self):
        out = None
        for content in self._contents:
            if out is None:
                out = content.purelist_depth
            elif out != content.purelist_depth:
                return -1
        else:
            return out

    @property
    def minmax_depth(self):
        if len(self._contents) == 0:
            return (0, 0)
        mins, maxs = [], []
        for content in self._contents:
            mindepth, maxdepth = content.minmax_depth
            mins.append(mindepth)
            maxs.append(maxdepth)
        return (min(mins), max(maxs))

    @property
    def branch_depth(self):
        if len(self._contents) == 0:
            return (False, 1)
        anybranch = False
        mindepth = None
        for content in self._contents:
            branch, depth = content.branch_depth
            if mindepth is None:
                mindepth = depth
            if branch or mindepth != depth:
                anybranch = True
            if mindepth > depth:
                mindepth = depth
        return (anybranch, mindepth)

    @property
    def keys(self):
        keyslists = []
        for i in range(len(self._contents)):
            keyslists.append(self._contents[i].keys)
        return list(set.intersection(*[set(x) for x in keyslists]))
