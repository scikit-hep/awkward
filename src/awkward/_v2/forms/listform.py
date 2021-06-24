# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

from awkward._v2.forms.form import Form


class ListForm(Form):
    def __init__(
        self,
        starts,
        stops,
        content,
        has_identities=False,
        parameters=None,
        form_key=None,
    ):
        if not isinstance(starts, str):
            raise TypeError(
                "{0} 'starts' must be of type str, not {1}".format(
                    type(self).__name__, repr(starts)
                )
            )
        if not isinstance(stops, str):
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
        self._starts = starts
        self._stops = stops
        self._content = content
        self._has_identities = has_identities
        self._parameters = parameters
        self._form_key = form_key

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

    def _tolist_part(self, verbose=True, toplevel=False):
        out = {}
        out["class"] = "ListArray"
        out["starts"] = self._starts
        out["stops"] = self._stops
        out["content"] = self._content.tolist(verbose=verbose)
        return out
