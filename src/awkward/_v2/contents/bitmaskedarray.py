# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json
import copy
import math

import awkward as ak
from awkward._v2.index import Index
from awkward._v2._slicing import NestedIndexError
from awkward._v2.contents.content import Content
from awkward._v2.contents.bytemaskedarray import ByteMaskedArray
from awkward._v2.forms.bitmaskedform import BitMaskedForm
from awkward._v2.forms.form import _parameters_equal

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


class BitMaskedArray(Content):
    is_OptionType = True

    def __init__(
        self,
        mask,
        content,
        valid_when,
        length,
        lsb_order,
        identifier=None,
        parameters=None,
    ):
        if not (isinstance(mask, Index) and mask.dtype == np.dtype(np.uint8)):
            raise TypeError(
                "{0} 'mask' must be an Index with dtype=uint8, not {1}".format(
                    type(self).__name__, repr(mask)
                )
            )
        if not isinstance(content, Content):
            raise TypeError(
                "{0} 'content' must be a Content subtype, not {1}".format(
                    type(self).__name__, repr(content)
                )
            )
        if not isinstance(valid_when, bool):
            raise TypeError(
                "{0} 'valid_when' must be boolean, not {1}".format(
                    type(self).__name__, repr(valid_when)
                )
            )
        if not ak._util.isint(length):
            raise TypeError(
                "{0} 'length' must be an integer, not {1}".format(
                    type(self).__name__, repr(length)
                )
            )
        if not isinstance(lsb_order, bool):
            raise TypeError(
                "{0} 'lsb_order' must be boolean, not {1}".format(
                    type(self).__name__, repr(lsb_order)
                )
            )
        if not length <= len(mask) * 8:
            raise ValueError(
                "{0} 'length' ({1}) must be <= len(mask) * 8 ({2})".format(
                    type(self).__name__, length, len(mask) * 8
                )
            )
        if not length <= len(content):
            raise ValueError(
                "{0} 'length' ({1}) must be <= len(content) ({2})".format(
                    type(self).__name__, length, len(content)
                )
            )

        self._mask = mask
        self._content = content
        self._valid_when = valid_when
        self._length = length
        self._lsb_order = lsb_order
        self._init(identifier, parameters)

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
    def nplike(self):
        return self._mask.nplike

    Form = BitMaskedForm

    def _form_with_key(self, getkey):
        form_key = getkey(self)
        return self.Form(
            self._mask.form,
            self._content._form_with_key(getkey),
            self._valid_when,
            self._lsb_order,
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=form_key,
        )

    def _to_buffers(self, form, getkey, container, nplike):
        assert isinstance(form, self.Form)
        key = getkey(self, form, "mask")
        container[key] = ak._v2._util.little_endian(self._mask.to(nplike))
        self._content._to_buffers(form.content, getkey, container, nplike)

    @property
    def typetracer(self):
        return BitMaskedArray(
            ak._v2.index.Index(self._mask.to(ak._v2._typetracer.TypeTracer.instance())),
            self._content.typetracer,
            self._valid_when,
            self._length,
            self._lsb_order,
            self._typetracer_identifier(),
            self._parameters,
        )

    def __len__(self):
        return self._length

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<BitMaskedArray valid_when="]
        out.append(repr(json.dumps(self._valid_when)))
        out.append(" lsb_order=")
        out.append(repr(json.dumps(self._lsb_order)))
        out.append(" len=")
        out.append(repr(str(len(self))))
        out.append(">")
        out.extend(self._repr_extra(indent + "    "))
        out.append("\n")
        out.append(self._mask._repr(indent + "    ", "<mask>", "</mask>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent + "</BitMaskedArray>")
        out.append(post)
        return "".join(out)

    def merge_parameters(self, parameters):
        return BitMaskedArray(
            self._mask,
            self._content,
            self._valid_when,
            self._length,
            self._lsb_order,
            self._identifier,
            ak._v2._util.merge_parameters(self._parameters, parameters),
        )

    def toByteMaskedArray(self):
        nplike = self._mask.nplike
        bytemask = ak._v2.index.Index8.empty(len(self._mask) * 8, nplike)
        self._handle_error(
            nplike[
                "awkward_BitMaskedArray_to_ByteMaskedArray",
                bytemask.dtype.type,
                self._mask.dtype.type,
            ](
                bytemask.to(nplike),
                self._mask.to(nplike),
                len(self._mask),
                False,  # this differs from the kernel call in 'bytemask'
                self._lsb_order,
            )
        )
        return ByteMaskedArray(
            bytemask[: self._length],
            self._content,
            self._valid_when,
            self._identifier,
            self._parameters,
        )

    def toIndexedOptionArray64(self):
        nplike = self._mask.nplike
        index = ak._v2.index.Index64.empty(len(self._mask) * 8, nplike)
        self._handle_error(
            nplike[
                "awkward_BitMaskedArray_to_IndexedOptionArray",
                index.dtype.type,
                self._mask.dtype.type,
            ](
                index.to(nplike),
                self._mask.to(nplike),
                len(self._mask),
                self._valid_when,
                self._lsb_order,
            ),
        )
        return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            index[0 : self._length],
            self._content,
            self._identifier,
            self._parameters,
        )

    def mask_as_bool(self, valid_when=None, nplike=None):
        if valid_when is None:
            valid_when = self._valid_when
        if nplike is None:
            nplike = self._mask.nplike

        bytemask = ak._v2.index.Index8.empty(len(self._mask) * 8, nplike)
        self._handle_error(
            nplike[
                "awkward_BitMaskedArray_to_ByteMaskedArray",
                bytemask.dtype.type,
                self._mask.dtype.type,
            ](
                bytemask.to(nplike),
                self._mask.to(nplike),
                len(self._mask),
                0 if valid_when == self._valid_when else 1,
                self._lsb_order,
            )
        )
        return bytemask.data[: self._length].view(np.bool_)

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if not (0 <= where < len(self)) and self.nplike.known_shape:
            raise NestedIndexError(self, where)
        if self._lsb_order:
            bit = bool(self._mask[where // 8] & (1 << (where % 8)))
        else:
            bit = bool(self._mask[where // 8] & (128 >> (where % 8)))
        if bit == self._valid_when:
            return self._content._getitem_at(where)
        else:
            return None

    def _getitem_range(self, where):
        return self.toByteMaskedArray()._getitem_range(where)

    def _getitem_field(self, where, only_fields=()):
        return BitMaskedArray(
            self._mask,
            self._content._getitem_field(where, only_fields),
            self._valid_when,
            self._length,
            self._lsb_order,
            self._field_identifier(where),
            None,
        )

    def _getitem_fields(self, where, only_fields=()):
        return BitMaskedArray(
            self._mask,
            self._content._getitem_fields(where, only_fields),
            self._valid_when,
            self._length,
            self._lsb_order,
            self._fields_identifier(where),
            None,
        )

    def _carry(self, carry, allow_lazy, exception):
        assert isinstance(carry, ak._v2.index.Index)
        return self.toByteMaskedArray()._carry(carry, allow_lazy, exception)

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        return self.toByteMaskedArray()._getitem_next_jagged(
            slicestarts, slicestops, slicecontent, tail
        )

    def _getitem_next(self, head, tail, advanced):
        if head == ():
            return self

        elif isinstance(head, (int, slice, ak._v2.index.Index64)):
            return self.toByteMaskedArray()._getitem_next(head, tail, advanced)

        elif ak._util.isstr(head):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, ak._v2.contents.ListOffsetArray):
            return self.toByteMaskedArray()._getitem_next(head, tail, advanced)

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise AssertionError(repr(head))

    def project(self, mask=None):
        return self.toByteMaskedArray().project(mask)

    def simplify_optiontype(self):
        if isinstance(
            self._content,
            (
                ak._v2.contents.indexedarray.IndexedArray,
                ak._v2.contents.indexedoptionarray.IndexedOptionArray,
                ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                ak._v2.contents.bitmaskeddarray.BitMaskedArray,
                ak._v2.contents.unmaskeddarray.UnmaskedArray,
            ),
        ):
            return self.toIndexedOptionArray64.simplify_optiontype
        else:
            return self

    def num(self, axis, depth=0):
        return self.toByteMaskedArray.num(axis, depth)

    def _offsets_and_flattened(self, axis, depth):
        return self.toByteMaskedArray._offsets_and_flattened(axis, depth)

    def mergeable(self, other, mergebool):
        if not _parameters_equal(self._parameters, other._parameters):
            return False

        if isinstance(
            other,
            (
                ak._v2.contents.emptyArray.EmptyArray,
                ak._v2.contents.unionarray.UnionArray,
            ),
        ):
            return True

        if isinstance(
            other,
            (
                ak._v2.contents.indexedarray.IndexedArray,
                ak._v2.contents.indexedoptionarray.IndexedOptionArray,
                ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                ak._v2.contents.bitmaskedarray.BitMaskedArray,
                ak._v2.contents.unmaskedarray.UnmaskedArray,
            ),
        ):
            self._content.mergeable(other.content, mergebool)

        else:
            return self._content.mergeable(other, mergebool)

    def _reverse_merge(self, other):
        return self.toIndexedOptionArray64()._reverse_merge(other)

    def mergemany(self, others):
        if len(others) == 0:
            return self
        return self.toIndexedOptionArray64().mergemany(others)

    def fillna(self, value):
        return self.toIndexedOptionArray64().fillna(value)

    def _localindex(self, axis, depth):
        return self.toByteMaskedArray()._localindex(axis, depth)

    def numbers_to_type(self, name):
        return self.toByteMaskedArray().numbers_to_type(name)

    def _is_unique(self, negaxis, starts, parents, outlength):
        if len(self._mask) == 0:
            return True
        return self.toIndexedOptionArray64()._is_unique(
            negaxis, starts, parents, outlength
        )

    def _unique(self, negaxis, starts, parents, outlength):
        if len(self._mask) == 0:
            return self
        out = self.toIndexedOptionArray64()._unique(negaxis, starts, parents, outlength)
        if negaxis is None:
            return out
        else:
            return out._content

    def _argsort_next(
        self,
        negaxis,
        starts,
        shifts,
        parents,
        outlength,
        ascending,
        stable,
        kind,
        order,
    ):
        if len(self._mask) == 0:
            return ak._v2.contents.NumpyArray(self.nplike.empty(0, np.int64))

        return self.toIndexedOptionArray64()._argsort_next(
            negaxis,
            starts,
            shifts,
            parents,
            outlength,
            ascending,
            stable,
            kind,
            order,
        )

    def _sort_next(
        self, negaxis, starts, parents, outlength, ascending, stable, kind, order
    ):
        return self.toIndexedOptionArray64()._sort_next(
            negaxis,
            starts,
            parents,
            outlength,
            ascending,
            stable,
            kind,
            order,
        )

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        return self.toByteMaskedArray()._combinations(
            n, replacement, recordlookup, parameters, axis, depth
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
        return self.toByteMaskedArray()._reduce_next(
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
        if len(self.mask) * 8 < len(self):
            return 'at {0} ("{1}"): len(mask) * 8 < length'.format(path, type(self))
        elif len(self._content) < len(self):
            return 'at {0} ("{1}"): len(content) < length'.format(path, type(self))
        elif isinstance(
            self._content,
            (
                ak._v2.contents.bitmaskedarray.BitMaskedArray,
                ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                ak._v2.contents.indexedarray.IndexedArray,
                ak._v2.contents.indexedoptionarray.IndexedOptionArray,
                ak._v2.contents.unmaskedarray.UnmaskedArray,
            ),
        ):
            return "{0} contains \"{1}\", the operation that made it might have forgotten to call 'simplify_optiontype()'"
        else:
            return self._content.validityerror(path + ".content")

    def _nbytes_part(self):
        result = self.mask._nbytes_part() + self.content._nbytes_part()
        if self.identifier is not None:
            result = result + self.identifier._nbytes_part()
        return result

    def _rpad(self, target, axis, depth, clip):
        return self.toByteMaskedArray()._rpad(target, axis, depth, clip)

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        return self.toByteMaskedArray()._to_arrow(
            pyarrow, mask_node, validbytes, length, options
        )

    def _to_numpy(self, allow_missing):
        return self.toByteMaskedArray()._to_numpy(allow_missing)

    def _completely_flatten(self, nplike, options):
        return self.project()._completely_flatten(nplike, options)

    def _recursively_apply(
        self, action, depth, depth_context, lateral_context, options
    ):
        if options["return_array"]:

            def continuation():
                return BitMaskedArray(
                    self._mask,
                    self._content._recursively_apply(
                        action,
                        depth,
                        copy.copy(depth_context),
                        lateral_context,
                        options,
                    ),
                    self._valid_when,
                    self._length,
                    self._lsb_order,
                    self._identifier,
                    self._parameters if options["keep_parameters"] else None,
                )

        else:

            def continuation():
                self._content._recursively_apply(
                    action,
                    depth,
                    copy.copy(depth_context),
                    lateral_context,
                    options,
                )

        result = action(
            self,
            depth=depth,
            depth_context=depth_context,
            lateral_context=lateral_context,
            continuation=continuation,
            options=options,
        )

        if isinstance(result, Content):
            return result
        elif result is None:
            return continuation()
        else:
            raise AssertionError(result)

    def packed(self):
        if self._content.is_RecordType:
            next = self.toIndexedOptionArray64()

            content = next._content.packed()
            if len(content) > self._length:
                content = content[: self._length]

            return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                next._index,
                content,
                next._identifier,
                next._parameters,
            )

        else:
            excess_length = int(math.ceil(self._length / 8.0))
            if len(self._mask) == excess_length:
                mask = self._mask
            else:
                mask = self._mask[:excess_length]

            content = self._content.packed()
            if len(content) > self._length:
                content = content[: self._length]

            return BitMaskedArray(
                mask,
                content,
                self._valid_when,
                self._length,
                self._lsb_order,
                self._identifier,
                self._parameters,
            )

    def _to_list(self, behavior):
        out = self._to_list_custom(behavior)
        if out is not None:
            return out

        mask = self.mask_as_bool(valid_when=True, nplike=numpy)[: self._length]
        content = self._content._to_list(behavior)
        out = [None] * self._length
        for i, isvalid in enumerate(mask):
            if isvalid:
                out[i] = content[i]
        return out
