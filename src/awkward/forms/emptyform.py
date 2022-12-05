# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
from awkward._util import unset
from awkward.forms.form import Form


class EmptyForm(Form):
    is_numpy = True
    is_unknown = True

    def __init__(self, *, parameters=None, form_key=None):
        self._init(parameters, form_key)

    def copy(self, *, parameters=unset, form_key=unset):
        return EmptyForm(parameters=parameters, form_key=form_key)

    @classmethod
    def simplified(cls, *, parameters=None, form_key=None):
        return cls(parameters=parameters, form_key=form_key)

    def __repr__(self):
        args = self._repr_args()
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _to_dict_part(self, verbose, toplevel):
        return self._to_dict_extra({"class": "EmptyArray"}, verbose)

    def _type(self, typestrs):
        return ak.types.UnknownType(
            parameters=self._parameters,
            typestr=ak._util.gettypestr(self._parameters, typestrs),
        )

    def __eq__(self, other):
        return isinstance(other, EmptyForm) and self._form_key == other._form_key

    def to_NumpyForm(self, dtype):
        return ak.forms.numpyform.from_dtype(dtype, parameters=self._parameters)

    def purelist_parameter(self, key):
        if self._parameters is None or key not in self._parameters:
            return None
        else:
            return self._parameters[key]

    @property
    def purelist_isregular(self):
        return True

    @property
    def purelist_depth(self):
        return 1

    @property
    def is_identity_like(self):
        return True

    @property
    def minmax_depth(self):
        return (1, 1)

    @property
    def branch_depth(self):
        return (False, 1)

    @property
    def fields(self):
        return []

    @property
    def is_tuple(self):
        return False

    @property
    def dimension_optiontype(self):
        return False

    def _columns(self, path, output, list_indicator):
        output.append(".".join(path))

    def _select_columns(self, index, specifier, matches, output):
        if any(match and index >= len(item) for item, match in zip(specifier, matches)):
            output.append(None)
        return self

    def _column_types(self):
        return ("empty",)
