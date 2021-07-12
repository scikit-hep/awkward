# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.forms.form import Form


class BitMaskedForm(Form):
    def __init__(
        self,
        mask,
        content,
        valid_when,
        lsb_order,
        has_identifier=False,
        parameters=None,
        form_key=None,
    ):
        if not ak._util.isstr(mask):
            raise TypeError(
                "{0} 'mask' must be of type str, not {1}".format(
                    type(self).__name__, repr(mask)
                )
            )
        if not isinstance(content, Form):
            raise TypeError(
                "{0} all 'contents' must be Form subclasses, not {1}".format(
                    type(self).__name__, repr(content)
                )
            )
        if not isinstance(valid_when, bool):
            raise TypeError(
                "{0} 'valid_when' must be bool, not {1}".format(
                    type(self).__name__, repr(valid_when)
                )
            )
        if not isinstance(lsb_order, bool):
            raise TypeError(
                "{0} 'lsb_order' must be bool, not {1}".format(
                    type(self).__name__, repr(lsb_order)
                )
            )

        self._mask = mask
        self._content = content
        self._valid_when = valid_when
        self._lsb_order = lsb_order
        self._init(has_identifier, parameters, form_key)

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

    def __repr__(self):
        args = [
            repr(self._mask),
            repr(self._content),
            repr(self._valid_when),
            repr(self._lsb_order),
        ] + self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def _tolist_part(self, verbose, toplevel):
        return self._tolist_extra(
            {
                "class": "BitMaskedArray",
                "mask": self._mask,
                "valid_when": self._valid_when,
                "lsb_order": self._lsb_order,
                "content": self._content._tolist_part(verbose, toplevel=False),
            },
            verbose,
        )
