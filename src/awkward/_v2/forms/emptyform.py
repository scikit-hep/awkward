# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

from awkward._v2.forms.form import Form


class EmptyForm(Form):
    def __init__(self, has_identifier=False, parameters=None, form_key=None):
        self._init(has_identifier, parameters, form_key)

    def __repr__(self):
        args = self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def _tolist_part(self, verbose, toplevel):
        return self._tolist_extra({"class": "EmptyArray"}, verbose)
