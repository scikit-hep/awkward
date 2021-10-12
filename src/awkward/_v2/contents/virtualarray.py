# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import numbers
import threading
import json

try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping

import awkward as ak
from awkward._v2.contents.content import Content, NestedIndexError
from awkward._v2.forms.form import Form, _parameters_update
from awkward._v2.forms.virtualform import VirtualForm

np = ak.nplike.NumpyMetadata.instance()


class ArrayGenerator(object):
    def __init__(self, length, form):
        if length is not None and not isinstance(length, numbers.Integral):
            raise TypeError(
                "{0} 'length' must be None or an integer, not {1}".format(
                    type(self).__name__, repr(length)
                )
            )
        if form is not None and not isinstance(form, Form):
            raise TypeError(
                "{0} 'form' must be None or a Form, not {1}".format(
                    type(self).__name__, repr(form)
                )
            )
        self._length = length
        self._form = form

    @property
    def length(self):
        return self._length

    @property
    def form(self):
        return self._form

    @property
    def typetracer(self):
        generator = FunctionGenerator(
            self._generator.length,
            self._generator.form,
            lambda *a, **k: self._generator.function(*a, **k).typetracer,
            self._generator.args,
            self._generator.kwargs,
        )
        return VirtualArray(
            generator,
            None,
            self._cache_key,
            self._typetracer_identifier(),
            self._parameters,
        )

    def generate(self):
        raise AssertionError()

    def generate_and_check(self):
        out = self.generate()
        # FIXME: do to_layout here

        if self._length is not None:
            if self._length > len(out):
                raise ValueError(
                    "generated array does not have sufficient length: expected {0} but generated {1}".format(
                        self._length, len(out)
                    )
                )
            elif self._length < len(out):
                out = out[: self._length]

        generated_form = out.form
        if self._form is not None and not self._form.generated_compatibility(
            generated_form
        ):
            if generated_form is None:
                generated_str = "null"
            else:
                generated_str = json.dumps(generated_form.tolist(False), indent=4)
            expected_str = json.dumps(self._form.tolist(False), indent=4)
            generated_strs = [x.ljust(38)[:38] for x in generated_str.split("\n")]
            expected_strs = expected_str.split("\n")
            if len(generated_strs) < len(expected_strs):
                generated_strs.extend([""] * (len(expected_strs) - len(generated_strs)))
            if len(expected_strs) < len(generated_strs):
                expected_strs.extend([""] * (len(generated_strs) - len(expected_strs)))
            two_column = [
                x + ("   " if x.strip() == y.strip() else " | ") + y
                for x, y in zip(generated_strs, expected_strs)
            ]
            header1 = "generated                                expected"
            header2 = "-" * 79
            raise ValueError(
                "generated array does not conform to expected Form\n\n{0}\n{1}\n{2}".format(
                    header1, header2, "\n".join(two_column)
                )
            )

        return out


class FunctionGenerator(ArrayGenerator):
    def __init__(self, length, form, function, args=None, kwargs=None):
        super(FunctionGenerator, self).__init__(length, form)
        if not callable(function):
            raise TypeError(
                "{0} 'function' must be callable, not {1}".format(
                    type(self).__name__, repr(function)
                )
            )
        if args is not None and not isinstance(args, tuple):
            raise TypeError(
                "{0} 'args' must be None or a tuple, not {1}".format(
                    type(self).__name__, repr(args)
                )
            )
        if kwargs is not None and not isinstance(kwargs, dict):
            raise TypeError(
                "{0} 'kwargs' must be None or a dict, not {1}".format(
                    type(self).__name__, repr(kwargs)
                )
            )
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        self._function = function
        self._args = args
        self._kwargs = kwargs

    @property
    def function(self):
        return self._function

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<FunctionGenerator"]
        if len(self._args) != 0:
            out.append(" args=")
            out.append(repr(self._args))
        if len(self._kwargs) != 0:
            out.append(" kwargs=")
            out.append(repr(self._kwargs))
        out.append("><![CDATA[\n" + indent + "    ")
        out.append(repr(self._function))
        out.append("\n" + indent + "]]></FunctionGenerator>")
        return "".join(out)

    def generate(self):
        return self._function(*self._args, **self._kwargs)


class SliceGenerator(ArrayGenerator):
    def __init__(self, length, form, virtualarray, slice):
        super(SliceGenerator, self).__init__(length, form)
        if not isinstance(virtualarray, VirtualArray):
            raise TypeError(
                "{0} 'virtualarray' must be a VirtualArray, not {1}".format(
                    type(self).__name__, repr(virtualarray)
                )
            )
        self._virtualarray = virtualarray
        self._slice = slice

    @property
    def virtualarray(self):
        return self._virtualarray

    @property
    def slice(self):
        return self._slice

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<SliceGenerator>\n"]
        out.append(indent + "    <slice>")
        out.append(str(self._slice).replace("\n", "\n" + indent + "        "))
        out.append("</slice>\n")
        out.append(self._virtualarray._repr(indent + "    ", "", "", detailed=False))
        out.append("\n" + indent + "</SliceGenerator>")
        return "".join(out)

    def generate(self):
        return self._virtualarray.array[self._slice]


def _parameters_nonvirtual(form):
    if form is None:
        return None

    elif isinstance(form, ak._v2.forms.virtualform.VirtualForm):
        p = _parameters_nonvirtual(form.form)
        if p is None and form._parameters is None:
            return None
        elif p is None:
            return form._parameters
        elif form._parameters is None:
            return p
        else:
            p2 = p.copy()
            _parameters_update(p2, form._parameters)
            return p2

    else:
        return form._parameters


class VirtualArray(Content):
    _new_key_lock = threading.Lock()
    _new_key_number = 0

    @staticmethod
    def new_key():
        with VirtualArray._new_key_lock:
            out = VirtualArray._new_key_number
            VirtualArray._new_key_number += 1
        return "ak" + str(out)

    def __init__(
        self, generator, cache=None, cache_key=None, identifier=None, parameters=None
    ):
        if not isinstance(generator, ArrayGenerator):
            raise TypeError(
                "{0} 'generator' must be an ArrayGenerator, not {1}".format(
                    type(self).__name__, repr(generator)
                )
            )
        if cache is not None and not isinstance(cache, MutableMapping):
            raise TypeError(
                "{0} 'cache' must be None or a MutableMapping, not {1}".format(
                    type(self).__name__, repr(cache)
                )
            )
        if cache_key is not None and not ak._util.isstr(cache_key):
            raise TypeError(
                "{0} 'cache_key' must be None or a string, not {1}".format(
                    type(self).__name__, repr(cache_key)
                )
            )
        if cache_key is None:
            cache_key = VirtualArray.new_key()
        self._generator = generator
        self._cache = cache
        self._cache_key = cache_key

        p = _parameters_nonvirtual(generator.form)
        if p is None and parameters is None:
            self._init(identifier, None)
        elif p is None:
            self._init(identifier, parameters)
        elif parameters is None:
            self._init(identifier, p)
        else:
            p2 = p.copy()
            _parameters_update(p2, parameters)
            self._init(identifier, p2)

    Form = VirtualForm

    @property
    def generator(self):
        return self._generator

    @property
    def cache(self):
        return self._cache

    @property
    def cache_key(self):
        return self._cache_key

    @property
    def nplike(self):
        return self.array.nplike

    @property
    def nonvirtual_nplike(self):
        return None

    @property
    def form(self):
        return self.Form(
            self._generator.form,
            self._generator.length is not None,
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=self._cache_key,
        )

    def __len__(self):
        out = self._generator.length
        if out is None:
            return len(self.array)
        else:
            return out

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post, detailed=True):
        out = [indent, pre, "<VirtualArray cache_key="]
        out.append(repr(self._cache_key))
        out.append(" len=")
        out.append(repr(json.dumps(self._generator.length)))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._generator._repr(indent + "    ", "", ""))
        if detailed:
            out.append("\n" + indent + "    <form>\n" + indent + "        ")
            out.append(
                str(self._generator.form).replace("\n", "\n" + indent + "        ")
            )
            out.append("\n" + indent + "    </form>\n")
            out.append(indent + "    <cache><![CDATA[\n")
            cache_repr = repr(self._cache)
            max_length = max(80 - len(indent) - 8 - 5, 40)
            if len(cache_repr) > max_length:
                half_length = max_length // 2
                cache_repr = (
                    cache_repr[: max_length - half_length]
                    + " ... "
                    + cache_repr[-half_length:]
                )
            out.append(indent + "        " + cache_repr)
            out.append("\n" + indent + "    ]]></cache>\n")
        else:
            out.append("\n")
        out.append(indent + "</VirtualArray>")
        out.append(post)
        return "".join(out)

    @property
    def peek_array(self):
        if self._cache is not None:
            return self._cache.get(self._cache_key)
        else:
            return None

    @property
    def array(self):
        out = None
        if self._cache is not None:
            out = self._cache.get(self._cache_key)
        if out is None:
            out = self._generator.generate_and_check()
            if self._cache is not None:
                self._cache[self._cache_key] = out
        return out

    def _getitem_nothing(self):
        return self.array._getitem_nothing()

    def _getitem_at(self, where):
        return self.array._getitem_at(where)

    def _getitem_range(self, where):
        assert where.step is None or where.step == 1

        peek = self.peek_array
        if peek is not None:
            return peek._getitem_range(where)

        length = self._generator.length
        if length is None:
            return self.array._getitem_range(where)

        start, stop, _ = where.indices(length)

        if self._generator.form is None:
            form = None
        else:
            form = self._generator.form._getitem_range()

        start, stop, _ = where.indices(length)
        return VirtualArray(
            SliceGenerator(stop - start, form, self, where),
            cache=self._cache,
            cache_key=None,
            identifier=self._range_identifier(start, stop),
            parameters=self._parameters,
        )

    def _getitem_field(self, where, only_fields=()):
        peek = self.peek_array
        if peek is not None:
            return peek._getitem_field(where, only_fields)

        if self._generator.form is None:
            form = None
        else:
            form = self._generator.form._getitem_field(where, only_fields)

        return VirtualArray(
            SliceGenerator(self._generator.length, form, self, where),
            cache=self._cache,
            cache_key=None,
            identifier=self._field_identifier(where),
            parameters=self._parameters,
        )

    def _getitem_fields(self, where, only_fields=()):
        peek = self.peek_array
        if peek is not None:
            return peek._getitem_fields(where, only_fields)

        if self._generator.form is None:
            form = None
        else:
            form = self._generator.form._getitem_fields(where, only_fields)

        return VirtualArray(
            SliceGenerator(self._generator.length, form, self, where),
            cache=self._cache,
            cache_key=None,
            identifier=self._field_identifier(where),
            parameters=self._parameters,
        )

    def _carry(self, carry, allow_lazy, exception):
        assert isinstance(carry, ak._v2.index.Index)

        peek = self.peek_array
        if peek is not None:
            return peek._carry(carry, allow_lazy, exception)

        if self._generator.form is None:
            form = None
        else:
            form = self._generator.form._carry(allow_lazy)

        return VirtualArray(
            SliceGenerator(len(carry), form, self, carry.data),
            cache=self._cache,
            cache_key=None,
            identifier=self._carry_identifier(carry, NestedIndexError),
            parameters=self._parameters,
        )

    def _getitem_next(self, head, tail, advanced):
        return self.array._getitem_next(head, tail, advanced)

    def _localindex(self, axis, depth):
        posaxis = self._axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._localindex_axis0()
        else:
            return self.array._localindex(posaxis, depth)

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = self._axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        else:
            return self.array._combinations(
                n, replacement, recordlookup, parameters, posaxis, depth
            )

    def _reduce_next(
        self,
        reducer,
        negaxis,
        starts,
        shifts,
        parents,
        outlength,
        mask,
        keepdims,
    ):
        return self.array._reduce_next(
            reducer,
            negaxis,
            starts,
            shifts,
            parents,
            outlength,
            mask,
            keepdims,
        )

    def _validityerror(self, path):
        return NotImplementedError
