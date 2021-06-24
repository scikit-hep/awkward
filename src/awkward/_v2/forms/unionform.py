# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from awkward._v2.forms.form import Form


class UnionForm(Form):
    def __init__(
        self,
        tags,
        index,
        contents,
        has_identities=False,
        parameters=None,
        form_key=None,
    ):
        if not isinstance(tags, str):
            raise TypeError(
                "{0} 'tags' must be of type str, not {1}".format(
                    type(self).__name__, repr(tags)
                )
            )
        if not isinstance(index, str):
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
        if not isinstance(contents, list):
            contents = list(contents)
        for content in contents:
            if not isinstance(content, Form):
                raise TypeError(
                    "{0} all 'contents' must be Form subclasses, not {1}".format(
                        type(self).__name__, repr(content)
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
        self._tags = tags
        self._index = index
        self._contents = contents
        self._has_identities = has_identities
        self._parameters = parameters
        self._form_key = form_key

    @property
    def tags(self):
        return self._tags

    @property
    def index(self):
        return self._index

    @property
    def contents(self):
        return self._contents

    def __repr__(self):
        args = [
            repr(self._tags),
            repr(self._index),
            repr(self._contents),
        ] + self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def _tolist_part(self, verbose=True, toplevel=False):
        out = {}
        out["class"] = "UnionArray"
        out["tags"] = self._tags
        out["index"] = self._index
        contents_tolist = [self._contents[0].tolist(verbose=verbose)]
        contents_tolist += [
            content.tolist(verbose=verbose, toplevel=not verbose)
            for content in self._contents[1:]
        ]
        out["contents"] = contents_tolist
        return out
