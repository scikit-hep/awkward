# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

from awkward._v2.forms.form import Form, parameters_equal


class EmptyForm(Form):
    def __init__(self, has_identifier=False, parameters=None, form_key=None):
        self._init(has_identifier, parameters, form_key)

    def __repr__(self):
        args = self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def _tolist_part(self, verbose, toplevel):
        return self._tolist_extra({"class": "EmptyArray"}, verbose)

    def generated_compatibility(self, layout):
        from awkward._v2.contents.emptyarray import EmptyArray

        if isinstance(layout, EmptyArray):
            return parameters_equal(self._parameters, layout.parameters)
        else:
            return False

    @property
    def purelist_isregular(self):
        return True

    @property
    def purelist_depth(self):
        return 1

    @property
    def minmax_depth(self):
        return (1, 1)

    @property
    def branch_depth(self):
        return (False, 1)

    @property
    def keys(self):
        return []
