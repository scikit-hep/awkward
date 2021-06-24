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
        recordlookup,
        has_identities=False,
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
        if recordlookup is not None and not isinstance(recordlookup, Iterable):
            raise TypeError(
                "{0} 'recordlookup' must be iterable, not {1}".format(
                    type(self).__name__, repr(contents)
                )
            )

        self._recordlookup = recordlookup
        self._contents = list(contents)
        self._init(has_identities, parameters, form_key)

    @property
    def recordlookup(self):
        return self._recordlookup

    @property
    def contents(self):
        return self._contents

    def __repr__(self):
        args = [repr(self._contents), repr(self._recordlookup)] + self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def _tolist_part(self, verbose, toplevel):
        out = {"class": "RecordArray"}
        contents_tolist = [
            content._tolist_part(verbose, toplevel=False) for content in self._contents
        ]
        if self._recordlookup is None:
            out["contents"] = contents_tolist
        else:
            out["contents"] = dict(zip(self._recordlookup, contents_tolist))
        return self._tolist_extra(out, verbose)
