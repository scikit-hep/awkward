# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

from awkward._v2.forms.form import Form


class RegularForm(Form):
    def __init__(
        self, content, size, has_identities=False, parameters=None, form_key=None
    ):
        if not isinstance(content, Form):
            raise TypeError(
                "{0} all 'contents' must be Form subclasses, not {1}".format(
                    type(self).__name__, repr(content)
                )
            )
        if not isinstance(size, int):
            raise TypeError(
                "{0} 'size' must be of type int, not {1}".format(
                    type(self).__name__, repr(size)
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
        self._content = content
        self._size = size
        self._has_identities = has_identities
        self._parameters = parameters
        self._form_key = form_key

    @property
    def content(self):
        return self._content

    @property
    def size(self):
        return self._size

    def __repr__(self):
        args = [repr(self._content), repr(self._size)] + self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def _tolist_part(self, verbose=True, toplevel=False):
        out = {}
        out["class"] = "RegularArray"
        out["size"] = self._size
        out["content"] = self._content.tolist(verbose=verbose)
        return out
