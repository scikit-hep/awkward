# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from awkward._v2.forms.form import Form

class RecordForm(Form):
    def __init__(
        self, contents, recordlookup, has_identities=False, parameters={}, form_key=None
    ):  
        if not isinstance(contents, Iterable):
            raise TypeError(
                "{0} 'contents' must be iterable, not {1}".format(
                    type(self).__name__, repr(contents)
                )
            )
        if not isinstance(contents, list):
            contents = list(contents)
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
        if has_identities is not None and not isinstance(has_identities, bool):
            raise TypeError(
                "{0} 'has_identities' must be of type bool or None, not {1}".format(
                    type(self).__name__, repr(has_identities)
                )
            )
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError(
                "{0} 'parameters' must be of type dict or None, not {1}".format(
                    type(self).__name__, repr(parameters)
                )
            )
        if form_key is not None and not isinstance(form_key, str):
            raise TypeError(
                "{0} 'form_key' must be of type string or None, not {1}".format(
                    type(self).__name__, repr(form_key)
                )
            )
        self._recordlookup = recordlookup
        self._contents = contents
        self._has_identities = has_identities
        self._parameters = parameters
        self._form_key = form_key

    @property
    def recordlookup(self):
        return self._recordlookup

    @property
    def contents(self):
        return self._contents

    def __repr__(self):
        args = [
            repr(self._recordlookup),
            repr(self._content),
        ] + self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def _tolist_part(self, verbose=True, ):
        out = {}
        out["class"] = "RecordArray"
        out["recordlookop"] = self._recordlookup
        out["content"] = [content.tolist(verbose=verbose) for content in self._contents]
        return out
