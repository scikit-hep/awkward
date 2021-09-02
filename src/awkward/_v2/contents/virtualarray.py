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
from awkward._v2.contents.content import Content
from awkward._v2.forms.form import Form
from awkward._v2.forms.virtualform import VirtualForm
from awkward._v2.tmp_for_testing import v1_to_v2

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

    def generate(self):
        raise AssertionError()

    def generate_and_check(self):
        out = self.generate()
        if self._length is not None and self._length > len(out):
            raise ValueError(
                "generated array does not have sufficient length: expected {0} but generated {1}".format(
                    self._length, len(out)
                )
            )
        if self._form is not None and not self._form.generated_compatibility(out):
            generated_form = out.form
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


class FunctionGenerator(ArrayGenerator):
    def __init__(self, length, form, function, args=None, kwargs=None):
        super(FunctionGenerator, self).__init__(length, form)
        if not isinstance(function, callable):
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
        out.append(">\n" + indent + "    ")
        out.append(repr(self._function))
        out.append("\n" + indent + "</FunctionGenerator>")
        return "".join(out)

    def generate(self):
        out = self._function(self._args, self._kwargs)
        return v1_to_v2(ak.to_layout(out, False, False))


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
        out.append(indent + "    ")
        out.append(self._virtualarray._repr(indent + "        ", "", ""))
        out.append("\n" + indent + "</SliceGenerator>")
        return "".join(out)

    def generate(self):
        return self._virtualarray.array[self._slice]


class VirtualArray(Content):
    _new_key_lock = threading.Lock()
    _new_key_number = 0

    @staticmethod
    def new_key():
        with VirtualArray._newkey_lock:
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
        self._init(identifier, parameters)

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
    def form(self):
        return self.Form(
            self._generator.form,
            self._generator.length is not None,
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=self._form_key,
        )

    def __len__(self):
        out = self._generator.length
        if out is None:
            return len(self.array)
        else:
            return out

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<VirtualArray cache_key="]
        out.append(repr(self._cache_key))
        out.append(" len=")
        out.append(repr(json.dumps(self._generator.length)))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._generator._repr(indent + "    ", "", ""))
        out.append(indent + "    <form>")
        out.append(str(self._generator.form).replace("\n", "\n" + indent + "        "))
        out.append("</form>\n")
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
        return out

    def _getitem_nothing(self):
        return self.array._getitem_nothing()

    def _getitem_at(self, where):
        return self.array._getitem_at(where)

    def _getitem_range(self, where):
        assert where.step is None or where.step == 1

        length = self._generator.length
        if length is None:
            return self.array._getitem_range(where)

        start, stop, _ = where.indices(length)
        return VirtualArray(
            SliceGenerator(stop - start, self._generator.form, self, where),
            cache=self._cache,
            cache_key=None,
            identifier=None,
            parameters=self._parameters,
        )


#     def _getitem_field(self, where, only_fields=()):
#         return IndexedArray(
#             self._index,
#             self._content._getitem_field(where, only_fields),
#             self._field_identifier(where),
#             None,
#         )

#     def _getitem_fields(self, where, only_fields=()):
#         return IndexedArray(
#             self._index,
#             self._content._getitem_fields(where, only_fields),
#             self._fields_identifier(where),
#             None,
#         )

#     def _carry(self, carry, allow_lazy, exception):
#         assert isinstance(carry, ak._v2.index.Index)

#         try:
#             nextindex = self._index[carry.data]
#         except IndexError as err:
#             if issubclass(exception, NestedIndexError):
#                 raise exception(self, carry.data, str(err))
#             else:
#                 raise exception(str(err))

#         return IndexedArray(
#             nextindex,
#             self._content,
#             self._carry_identifier(carry, exception),
#             self._parameters,
#         )

#     def _getitem_next(self, head, tail, advanced):
#         nplike = self.nplike  # noqa: F841

#         if head == ():
#             return self

#         elif isinstance(head, (int, slice, ak._v2.index.Index64)):
#             nexthead, nexttail = self._headtail(tail)

#             nextcarry = ak._v2.index.Index64.empty(len(self._index), nplike)
#             self._handle_error(
#                 nplike[
#                     "awkward_IndexedArray_getitem_nextcarry",
#                     nextcarry.dtype.type,
#                     self._index.dtype.type,
#                 ](
#                     nextcarry.to(nplike),
#                     self._index.to(nplike),
#                     len(self._index),
#                     len(self._content),
#                 )
#             )

#             next = self._content._carry(nextcarry, False, NestedIndexError)
#             return next._getitem_next(head, tail, advanced)

#         elif ak._util.isstr(head):
#             return self._getitem_next_field(head, tail, advanced)

#         elif isinstance(head, list):
#             return self._getitem_next_fields(head, tail, advanced)

#         elif head is np.newaxis:
#             return self._getitem_next_newaxis(tail, advanced)

#         elif head is Ellipsis:
#             return self._getitem_next_ellipsis(tail, advanced)

#         elif isinstance(head, ak._v2.contents.ListOffsetArray):
#             raise NotImplementedError

#         elif isinstance(head, ak._v2.contents.IndexedOptionArray):
#             raise NotImplementedError

#         else:
#             raise AssertionError(repr(head))

#     def project(self):
#         nextcarry = ak._v2.index.Index64.empty(len(self), self.nplike)
#         self._handle_error(
#             self.nplike[
#                 "awkward_IndexedArray_getitem_nextcarry",
#                 nextcarry.dtype.type,
#                 self._index.dtype.type,
#             ](
#                 nextcarry.to(self.nplike),
#                 self._index.to(self.nplike),
#                 len(self._index),
#                 len(self._content),
#             )
#         )
#         return self._content._carry(nextcarry, False, NestedIndexError)

#     def _localindex(self, axis, depth):
#         posaxis = self._axis_wrap_if_negative(axis)
#         if posaxis == depth:
#             return self._localindex_axis0()
#         else:
#             return self.project()._localindex(posaxis, depth)
