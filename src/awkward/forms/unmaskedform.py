# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
from awkward._util import unset
from awkward.forms.form import Form, _parameters_equal


class UnmaskedForm(Form):
    is_option = True

    def __init__(
        self,
        content,
        *,
        parameters=None,
        form_key=None,
    ):
        if not isinstance(content, Form):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} all 'contents' must be Form subclasses, not {}".format(
                        type(self).__name__, repr(content)
                    )
                )
            )

        self._content = content
        self._init(parameters, form_key)

    @property
    def content(self):
        return self._content

    def copy(
        self,
        content=unset,
        *,
        parameters=unset,
        form_key=unset,
    ):
        return UnmaskedForm(
            self._content if content is unset else content,
            parameters=self._parameters if parameters is unset else parameters,
            form_key=self._form_key if form_key is unset else form_key,
        )

    @classmethod
    def simplified(
        cls,
        content,
        *,
        parameters=None,
        form_key=None,
    ):
        if content.is_union:
            return content.copy(
                contents=[cls.simplified(x) for x in content.contents],
                parameters=ak._util.merge_parameters(content._parameters, parameters),
            )
        elif content.is_indexed or content.is_option:
            return content.copy(
                parameters=ak._util.merge_parameters(content._parameters, parameters)
            )
        else:
            return cls(content, parameters=parameters, form_key=form_key)

    def __repr__(self):
        args = [repr(self._content)] + self._repr_args()
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _to_dict_part(self, verbose, toplevel):
        return self._to_dict_extra(
            {
                "class": "UnmaskedArray",
                "content": self._content._to_dict_part(verbose, toplevel=False),
            },
            verbose,
        )

    def _type(self, typestrs):
        return ak.types.OptionType(
            self._content._type(typestrs),
            parameters=self._parameters,
            typestr=ak._util.gettypestr(self._parameters, typestrs),
        ).simplify_option_union()

    def __eq__(self, other):
        if isinstance(other, UnmaskedForm):
            return (
                self._form_key == other._form_key
                and _parameters_equal(
                    self._parameters, other._parameters, only_array_record=True
                )
                and self._content == other._content
            )
        else:
            return False

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
    def is_identity_like(self):
        return self._content.is_identity_like

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
        return UnmaskedForm(
            self._content._select_columns(index, specifier, matches, output),
            parameters=self._parameters,
            form_key=self._form_key,
        )

    def _column_types(self):
        return self._content._column_types()
