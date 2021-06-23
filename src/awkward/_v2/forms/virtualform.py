# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

from awkward._v2.forms.form import Form


class VirtualForm(Form):
    def __init__(
        self,
        form,
        has_length,
        has_identities=False,
        parameters=None,
        form_key=None,
    ):
        if form is not None and not isinstance(form, Form):
            raise TypeError(
                "{0} all 'form' must be Form subclasses, not {1}".format(
                    type(self).__name__, repr(form)
                )
            )
        if not isinstance(has_length, bool):
            raise TypeError(
                "{0} 'has_length' must be bool, not {1}".format(
                    type(self).__name__, repr(has_length)
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
        self._form = form
        self._has_length = has_length
        self._has_identities = has_identities
        self._parameters = parameters
        self._form_key = form_key

    @property
    def form(self):
        return self._form

    @property
    def has_length(self):
        return self._has_length

    def __repr__(self):
        args = [repr(self._form), repr(self._has_length)] + self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def _tolist_part(self, verbose=True, toplevel=False):
        out = {}
        out["class"] = "VirtualArray"
        out["form"] = (
            self._form.tolist(verbose=verbose) if self._form is not None else None
        )
        out["has_length"] = self._has_length
        return out
