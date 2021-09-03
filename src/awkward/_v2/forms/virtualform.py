# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

from awkward._v2.forms.form import Form, parameters_equal


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
                "{0} 'form' must be a Form instance, not {1}".format(
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

    def __eq__(self, other):
        if isinstance(other, VirtualForm):
            return (
                self._has_identifier == other._has_identifier
                and self._form_key == other._form_key
                and parameters_equal(self._parameters, other._parameters)
                and self._form == other._form
            )
        else:
            return False

    def generated_compatibility(self, layout):
        from awkward._v2.contents.virtualarray import VirtualArray

        if isinstance(layout, VirtualArray):
            if self._form is None or layout.form is None:
                return parameters_equal(self._parameters, layout._parameters)
            else:
                self_parameters = {}
                self_parameters.update(self._form.parameters)
                self_parameters.update(self.parameters)

                layout_parameters = {}
                layout_parameters.update(layout.form.parameters)
                layout_parameters.update(layout.parameters)

                return parameters_equal(
                    self_parameters, layout_parameters
                ) and self._form.generated_compatibility(layout.form)

        else:
            return False

    @property
    def purelist_isregular(self):
        if self._form is None:
            raise ValueError(
                "cannot determine the type of a virtual array without a Form"
            )
        else:
            return self._form.purelist_isregular

    @property
    def purelist_depth(self):
        if self._form is None:
            raise ValueError(
                "cannot determine the type of a virtual array without a Form"
            )
        else:
            return self._form.purelist_depth

    @property
    def minmax_depth(self):
        if self._form is None:
            raise ValueError(
                "cannot determine the type of a virtual array without a Form"
            )
        else:
            return self._form.minmax_depth

    @property
    def branch_depth(self):
        if self._form is None:
            raise ValueError(
                "cannot determine the type of a virtual array without a Form"
            )
        else:
            return self._form.branch_depth

    @property
    def keys(self):
        if self._form is None:
            raise ValueError(
                "cannot determine the type of a virtual array without a Form"
            )
        else:
            return self._form.keys
