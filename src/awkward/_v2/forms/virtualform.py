# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.forms.form import (
    Form,
    _parameters_equal,
    _parameters_update,
    nonvirtual,
)


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

    def _type(self, typestrs):
        if self._form is None:
            return ak._v2.types.unknowntype.UnknownType(
                self._parameters,
                ak._util.gettypestr(self._parameters, typestrs),
            )
        else:
            out = self._form._type(typestrs)

            if self._parameters is not None:
                if out._parameters is None:
                    out._parameters = self._parameters
                else:
                    out._parameters = dict(out._parameters)
                    _parameters_update(out._parameters, self._parameters)

            return out

    def __eq__(self, other):
        if isinstance(other, VirtualForm):
            return (
                self._has_identifier == other._has_identifier
                and self._form_key == other._form_key
                and _parameters_equal(self._parameters, other._parameters)
                and self._form == other._form
            )
        else:
            return False

    def generated_compatibility(self, other):
        if other is None:
            other_parameters = None
        else:
            other_parameters = other._parameters

        if not _parameters_equal(self._parameters, other_parameters):
            return False

        nonvirtual_self = nonvirtual(self)
        nonvirtual_other = nonvirtual(other)

        if nonvirtual_self is None or nonvirtual_other is None:
            return True
        else:
            return nonvirtual_self.generated_compatibility(nonvirtual_other)

    def _getitem_range(self):
        return VirtualForm(
            None if self._form is None else self._form._getitem_range(),
            True,
            has_identifier=self._has_identifier,
            parameters=self._parameters,
            form_key=None,
        )

    def _getitem_field(self, where, only_fields=()):
        return VirtualForm(
            None
            if self._form is None
            else self._form._getitem_field(where, only_fields),
            self._has_length,
            has_identifier=self._has_identifier,
            parameters=None,
            form_key=None,
        )

    def _getitem_fields(self, where, only_fields=()):
        return VirtualForm(
            None
            if self._form is None
            else self._form._getitem_fields(where, only_fields),
            self._has_length,
            has_identifier=self._has_identifier,
            parameters=None,
            form_key=None,
        )

    def _carry(self, allow_lazy):
        return VirtualForm(
            None if self._form is None else self._form._getitem_range(),
            True,
            has_identifier=self._has_identifier,
            parameters=self._parameters,
            form_key=None,
        )

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

    @property
    def dimension_optiontype(self):
        if self._form is None:
            raise ValueError(
                "cannot determine the type of a virtual array without a Form"
            )
        else:
            return self._form.dimension_optiontype
