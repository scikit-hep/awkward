# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
from awkward.forms.form import Form, _parameters_equal


class BitMaskedForm(Form):
    is_option = True

    def __init__(
        self,
        mask,
        content,
        valid_when,
        lsb_order,
        parameters=None,
        form_key=None,
    ):
        if not isinstance(mask, str):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'mask' must be of type str, not {}".format(
                        type(self).__name__, repr(mask)
                    )
                )
            )
        if not isinstance(content, Form):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} all 'contents' must be Form subclasses, not {}".format(
                        type(self).__name__, repr(content)
                    )
                )
            )
        if not isinstance(valid_when, bool):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'valid_when' must be bool, not {}".format(
                        type(self).__name__, repr(valid_when)
                    )
                )
            )
        if not isinstance(lsb_order, bool):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'lsb_order' must be bool, not {}".format(
                        type(self).__name__, repr(lsb_order)
                    )
                )
            )

        self._mask = mask
        self._content = content
        self._valid_when = valid_when
        self._lsb_order = lsb_order
        self._init(parameters, form_key)

    @property
    def mask(self):
        return self._mask

    @property
    def content(self):
        return self._content

    @property
    def valid_when(self):
        return self._valid_when

    @property
    def lsb_order(self):
        return self._lsb_order

    @property
    def is_identity_like(self):
        return False

    def __repr__(self):
        args = [
            repr(self._mask),
            repr(self._content),
            repr(self._valid_when),
            repr(self._lsb_order),
        ] + self._repr_args()
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _to_dict_part(self, verbose, toplevel):
        return self._to_dict_extra(
            {
                "class": "BitMaskedArray",
                "mask": self._mask,
                "valid_when": self._valid_when,
                "lsb_order": self._lsb_order,
                "content": self._content._to_dict_part(verbose, toplevel=False),
            },
            verbose,
        )

    def _type(self, typestrs):
        return ak.types.OptionType(
            self._content._type(typestrs),
            self._parameters,
            typestr=ak._util.gettypestr(self._parameters, typestrs),
        ).simplify_option_union()

    def __eq__(self, other):
        if isinstance(other, BitMaskedForm):
            return (
                self._form_key == other._form_key
                and self._mask == other._mask
                and self._valid_when == other._valid_when
                and self._lsb_order == other._lsb_order
                and _parameters_equal(
                    self._parameters, other._parameters, only_array_record=True
                )
                and self._content == other._content
            )
        else:
            return False

    def simplify_optiontype(self):
        if isinstance(
            self._content,
            (
                ak.forms.IndexedForm,
                ak.forms.IndexedOptionForm,
                ak.forms.ByteMaskedForm,
                ak.forms.BitMaskedForm,
                ak.forms.UnmaskedForm,
            ),
        ):
            return ak.forms.IndexedOptionForm(
                "i64",
                self._content,
                parameters=self._parameters,
            ).simplify_optiontype()
        else:
            return self

    def purelist_parameter(self, key):
        if self._parameters is None or key not in self._parameters:
            return self._content.purelist_parameter(key)
        else:
            return self._parameters[key]

    @property
    def purelist_isregular(self):
        return self._content.purelist_isregular

    @property
    def purelist_depth(self):
        return self._content.purelist_depth

    @property
    def minmax_depth(self):
        return self._content.minmax_depth

    @property
    def branch_depth(self):
        return self._content.branch_depth

    @property
    def fields(self):
        return self._content.fields

    @property
    def is_tuple(self):
        return self._content.is_tuple

    @property
    def dimension_optiontype(self):
        return True

    def _columns(self, path, output, list_indicator):
        self._content._columns(path, output, list_indicator)

    def _select_columns(self, index, specifier, matches, output):
        return BitMaskedForm(
            self._mask,
            self._content._select_columns(index, specifier, matches, output),
            self._valid_when,
            self._lsb_order,
            self._parameters,
            self._form_key,
        )

    def _column_types(self):
        return self._content._column_types()
