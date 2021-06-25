# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

from awkward._v2.forms.form import Form


class VirtualForm(Form):
    def __init__(
        self,
        form,
        has_length,
        has_identifier=False,
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

        self._form = form
        self._has_length = has_length
        self._init(has_identifier, parameters, form_key)

    @property
    def form(self):
        return self._form

    @property
    def has_length(self):
        return self._has_length

    def __repr__(self):
        args = [repr(self._form), repr(self._has_length)] + self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def _tolist_part(self, verbose, toplevel):
        out = {"class": "VirtualArray"}
        if self._form is None:
            out["form"] = None
        else:
            out["form"] = self._form._tolist_part(verbose, toplevel=False)
        out["has_length"] = self._has_length
        return self._tolist_extra(out, verbose)
