# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from awkward._v2.forms.form import Form


class RecordForm(Form):
    def __init__(
        self,
        contents,
        keys,
        has_identifier=False,
        parameters=None,
        form_key=None,
    ):
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
        if keys is not None and not isinstance(keys, Iterable):
            raise TypeError(
                "{0} 'keys' must be iterable, not {1}".format(
                    type(self).__name__, repr(contents)
                )
            )

        self._keys = keys
        self._contents = list(contents)
        self._init(has_identifier, parameters, form_key)

    @property
    def keys(self):
        return self._keys

    @property
    def contents(self):
        return self._contents

    def __repr__(self):
        args = [repr(self._contents), repr(self._keys)] + self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def _tolist_part(self, verbose, toplevel):
        out = {"class": "RecordArray"}

        contents_tolist = [
            content._tolist_part(verbose, toplevel=False) for content in self._contents
        ]
        if self._keys is not None:
            out["contents"] = dict(zip(self._keys, contents_tolist))
        else:
            out["contents"] = contents_tolist

        return self._tolist_extra(out, verbose)

    @property
    def purelist_isregular(self):
        return True

    @property
    def purelist_depth(self):
        return 1

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
