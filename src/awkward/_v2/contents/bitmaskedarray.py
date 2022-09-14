# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import json
import copy
import math

import awkward as ak
from awkward._v2.index import Index
from awkward._v2.contents.content import Content, unset
from awkward._v2.contents.bytemaskedarray import ByteMaskedArray
from awkward._v2.forms.bitmaskedform import BitMaskedForm

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


class BitMaskedArray(Content):
    is_OptionType = True

    def copy(
        self,
        mask=unset,
        content=unset,
        valid_when=unset,
        length=unset,
        lsb_order=unset,
        identifier=unset,
        parameters=unset,
        nplike=unset,
    ):
        return BitMaskedArray(
            self._mask if mask is unset else mask,
            self._content if content is unset else content,
            self._valid_when if valid_when is unset else valid_when,
            self._length if length is unset else length,
            self._lsb_order if lsb_order is unset else lsb_order,
            self._identifier if identifier is unset else identifier,
            self._parameters if parameters is unset else parameters,
            self._nplike if nplike is unset else nplike,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(
            mask=copy.deepcopy(self._mask, memo),
            content=copy.deepcopy(self._content, memo),
            identifier=copy.deepcopy(self._identifier, memo),
            parameters=copy.deepcopy(self._parameters, memo),
        )

    def __init__(
        self,
        mask,
        content,
        valid_when,
        length,
        lsb_order,
        identifier=None,
        parameters=None,
        nplike=None,
    ):
        if not (isinstance(mask, Index) and mask.dtype == np.dtype(np.uint8)):
            raise ak._v2._util.error(
                TypeError(
                    "{} 'mask' must be an Index with dtype=uint8, not {}".format(
                        type(self).__name__, repr(mask)
                    )
                )
            )
        if not isinstance(content, Content):
            raise ak._v2._util.error(
                TypeError(
                    "{} 'content' must be a Content subtype, not {}".format(
                        type(self).__name__, repr(content)
                    )
                )
            )
        if not isinstance(valid_when, bool):
            raise ak._v2._util.error(
                TypeError(
                    "{} 'valid_when' must be boolean, not {}".format(
                        type(self).__name__, repr(valid_when)
                    )
                )
            )
        if not isinstance(length, ak._v2._typetracer.UnknownLengthType):
            if not (ak._util.isint(length) and length >= 0):
                raise ak._v2._util.error(
                    TypeError(
                        "{} 'length' must be a non-negative integer, not {}".format(
                            type(self).__name__, length
                        )
                    )
                )
        if not isinstance(lsb_order, bool):
            raise ak._v2._util.error(
                TypeError(
                    "{} 'lsb_order' must be boolean, not {}".format(
                        type(self).__name__, repr(lsb_order)
                    )
                )
            )
        if length > mask.length * 8:
            raise ak._v2._util.error(
                ValueError(
                    "{} 'length' ({}) must be <= len(mask) * 8 ({})".format(
                        type(self).__name__, length, mask.length * 8
                    )
                )
            )
        if length > content.length:
            raise ak._v2._util.error(
                ValueError(
                    "{} 'length' ({}) must be <= len(content) ({})".format(
                        type(self).__name__, length, content.length
                    )
                )
            )
        if nplike is None:
            nplike = content.nplike
        if nplike is None:
            nplike = mask.nplike

        self._mask = mask
        self._content = content
        self._valid_when = valid_when
        self._length = length
        self._lsb_order = lsb_order
        self._init(identifier, parameters, nplike)

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
        container[key] = ak._v2._util.little_endian(self._mask.raw(nplike))
        self._content._to_buffers(form.content, getkey, container, nplike)

    @property
    def typetracer(self):
        tt = ak._v2._typetracer.TypeTracer.instance()
        return BitMaskedArray(
            ak._v2.index.Index(self._mask.raw(tt)),
            self._content.typetracer,
            self._valid_when,
            self._length,
            self._lsb_order,
            self._typetracer_identifier(),
            self._parameters,
            tt,
        )

    @property
    def length(self):
        return self._length

    def _forget_length(self):
        return BitMaskedArray(
            self._mask,
            self._content.typetracer,
            self._valid_when,
            ak._v2._typetracer.UnknownLength,
            self._lsb_order,
            self._identifier,
            self._parameters,
            self._nplike,
        )

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<BitMaskedArray valid_when="]
        out.append(repr(json.dumps(self._valid_when)))
        out.append(" lsb_order=")
        out.append(repr(json.dumps(self._lsb_order)))
        out.append(" len=")
        out.append(repr(str(self.length)))
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
            self._nplike,
        )

    def toIndexedOptionArray64(self):
        index = ak._v2.index.Index64.empty(self._mask.length * 8, self._nplike)
        assert index.nplike is self._nplike and self._mask.nplike is self._nplike
        self._handle_error(
            self._nplike[
                "awkward_BitMaskedArray_to_IndexedOptionArray",
                index.dtype.type,
                self._mask.dtype.type,
            ](
                index.raw(self._nplike),
                self._mask.data,
                self._mask.length,
                self._valid_when,
                self._lsb_order,
            ),
        )
        return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
            index[0 : self._length],
            self._content,
            self._identifier,
            self._parameters,
            self._nplike,
        )

    def toByteMaskedArray(self):
        bytemask = ak._v2.index.Index8.empty(self._mask.length * 8, self._nplike)
        assert bytemask.nplike is self._nplike and self._mask.nplike is self._nplike
        self._handle_error(
            self._nplike[
                "awkward_BitMaskedArray_to_ByteMaskedArray",
                bytemask.dtype.type,
                self._mask.dtype.type,
            ](
                bytemask.data,
                self._mask.data,
                self._mask.length,
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
            self._nplike,
        )

    def toBitMaskedArray(self, valid_when, lsb_order):
        if lsb_order == self._lsb_order:
            if valid_when == self._valid_when:
                return self
            else:
                return BitMaskedArray(
                    ak._v2.index.IndexU8(~self._mask.data),
                    self._content,
                    valid_when,
                    self._length,
                    lsb_order,
                    self._identifier,
                    self._parameters,
                    self._nplike,
                )

        else:
            import awkward._v2._connect.pyarrow

            bytemask = awkward._v2._connect.pyarrow.unpackbits(
                self._mask.data, self._lsb_order
            )
            bitmask = awkward._v2._connect.pyarrow.packbits(bytemask, lsb_order)

            if valid_when != self._valid_when:
                bitmask = ~bitmask

            return BitMaskedArray(
                ak._v2.index.IndexU8(bitmask),
                self._content,
                valid_when,
                self._length,
                lsb_order,
                self._identifier,
                self._parameters,
                self._nplike,
            )

    def mask_as_bool(self, valid_when=None, nplike=None):
        if valid_when is None:
            valid_when = self._valid_when
        if nplike is None:
            nplike = self._nplike

        bytemask = ak._v2.index.Index8.empty(self._mask.length * 8, nplike)
        assert bytemask.nplike is self._nplike and self._mask.nplike is self._nplike
        self._handle_error(
            nplike[
                "awkward_BitMaskedArray_to_ByteMaskedArray",
                bytemask.dtype.type,
                self._mask.dtype.type,
            ](
                bytemask.data,
                self._mask.data,
                self._mask.length,
                0 if valid_when == self._valid_when else 1,
                self._lsb_order,
            )
        )
        return bytemask.data[: self._length].view(np.bool_)

    def _getitem_nothing(self):
        return self._content._getitem_range(slice(0, 0))

    def _getitem_at(self, where):
        if not self._nplike.known_data:
            return ak._v2._typetracer.MaybeNone(self._content._getitem_at(where))

        if where < 0:
            where += self.length
        if not (0 <= where < self.length) and self._nplike.known_shape:
            raise ak._v2._util.indexerror(self, where)
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
            self._nplike,
        ).simplify_optiontype()

    def _getitem_fields(self, where, only_fields=()):
        return BitMaskedArray(
            self._mask,
            self._content._getitem_fields(where, only_fields),
            self._valid_when,
            self._length,
            self._lsb_order,
            self._fields_identifier(where),
            None,
            self._nplike,
        ).simplify_optiontype()

    def _carry(self, carry, allow_lazy):
        assert isinstance(carry, ak._v2.index.Index)
        return self.toByteMaskedArray()._carry(carry, allow_lazy)

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        return self.toByteMaskedArray()._getitem_next_jagged(
            slicestarts, slicestops, slicecontent, tail
        )

    def _getitem_next(self, head, tail, advanced):
        if head == ():
            return self

        elif isinstance(
            head, (int, slice, ak._v2.index.Index64, ak._v2.contents.ListOffsetArray)
        ):
            return self.toByteMaskedArray()._getitem_next(head, tail, advanced)

        elif ak._util.isstr(head):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif head is np.newaxis:
            return self._getitem_next_newaxis(tail, advanced)

        elif head is Ellipsis:
            return self._getitem_next_ellipsis(tail, advanced)

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            return self._getitem_next_missing(head, tail, advanced)

        else:
            raise ak._v2._util.error(AssertionError(repr(head)))

    def project(self, mask=None):
        return self.toByteMaskedArray().project(mask)

    def simplify_optiontype(self):
        if isinstance(
            self._content,
            (
                ak._v2.contents.indexedarray.IndexedArray,
                ak._v2.contents.indexedoptionarray.IndexedOptionArray,
                ak._v2.contents.bytemaskedarray.ByteMaskedArray,
                ak._v2.contents.bitmaskedarray.BitMaskedArray,
                ak._v2.contents.unmaskedarray.UnmaskedArray,
            ),
        ):
            return self.toIndexedOptionArray64().simplify_optiontype()
        else:
            return self

    def num(self, axis, depth=0):
        return self.toByteMaskedArray().num(axis, depth)

    def _offsets_and_flattened(self, axis, depth):
        return self.toByteMaskedArray._offsets_and_flattened(axis, depth)

    def _mergeable(self, other, mergebool):
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
            return self._content.mergeable(other.content, mergebool)

        else:
            return self._content.mergeable(other, mergebool)

    def _reverse_merge(self, other):
        return self.toIndexedOptionArray64()._reverse_merge(other)

    def mergemany(self, others):
        if len(others) == 0:
            return self

        out = self.toIndexedOptionArray64().mergemany(others)

        if all(
            isinstance(x, BitMaskedArray)
            and x._valid_when == self._valid_when
            and x._lsb_order == self._lsb_order
            for x in others
        ):
            return out.toBitMaskedArray(self._valid_when, self._lsb_order)
        else:
            return out

    def fill_none(self, value):
        return self.toIndexedOptionArray64().fill_none(value)

    def _local_index(self, axis, depth):
        return self.toByteMaskedArray()._local_index(axis, depth)

    def numbers_to_type(self, name):
        return self.toByteMaskedArray().numbers_to_type(name)

    def _is_unique(self, negaxis, starts, parents, outlength):
        if self._mask.length == 0:
            return True
        return self.toIndexedOptionArray64()._is_unique(
            negaxis, starts, parents, outlength
        )

    def _unique(self, negaxis, starts, parents, outlength):
        if self._mask.length == 0:
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
        behavior,
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
            behavior,
        )

    def _validity_error(self, path):
        if self.mask.length * 8 < self.length:
            return f'at {path} ("{type(self)}"): len(mask) * 8 < length'
        elif self._content.length < self.length:
            return f'at {path} ("{type(self)}"): len(content) < length'
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
            return self._content.validity_error(path + ".content")

    def _nbytes_part(self):
        result = self.mask._nbytes_part() + self.content._nbytes_part()
        if self.identifier is not None:
            result = result + self.identifier._nbytes_part()
        return result

    def _pad_none(self, target, axis, depth, clip):
        return self.toByteMaskedArray()._pad_none(target, axis, depth, clip)

    def _to_arrow(self, pyarrow, mask_node, validbytes, length, options):
        return self.toByteMaskedArray()._to_arrow(
            pyarrow, mask_node, validbytes, length, options
        )

    def _to_numpy(self, allow_missing):
        return self.toByteMaskedArray()._to_numpy(allow_missing)

    def _completely_flatten(self, nplike, options):
        return self.project()._completely_flatten(nplike, options)

    def _recursively_apply(
        self, action, behavior, depth, depth_context, lateral_context, options
    ):
        if self._nplike.known_shape:
            content = self._content[0 : self._length]
        else:
            content = self._content

        if options["return_array"]:

            def continuation():
                return BitMaskedArray(
                    self._mask,
                    content._recursively_apply(
                        action,
                        behavior,
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
                    self._nplike,
                )

        else:

            def continuation():
                content._recursively_apply(
                    action,
                    behavior,
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
            behavior=behavior,
            nplike=self._nplike,
            options=options,
        )

        if isinstance(result, Content):
            return result
        elif result is None:
            return continuation()
        else:
            raise ak._v2._util.error(AssertionError(result))

    def packed(self):
        if self._content.is_RecordType:
            next = self.toIndexedOptionArray64()

            content = next._content.packed()
            if content.length > self._length:
                content = content[: self._length]

            return ak._v2.contents.indexedoptionarray.IndexedOptionArray(
                next._index,
                content,
                next._identifier,
                next._parameters,
                self._nplike,
            )

        else:
            excess_length = int(math.ceil(self._length / 8.0))
            if self._mask.length == excess_length:
                mask = self._mask
            else:
                mask = self._mask[:excess_length]

            content = self._content.packed()
            if content.length > self._length:
                content = content[: self._length]

            return BitMaskedArray(
                mask,
                content,
                self._valid_when,
                self._length,
                self._lsb_order,
                self._identifier,
                self._parameters,
                self._nplike,
            )

    def _to_list(self, behavior, json_conversions):
        out = self._to_list_custom(behavior, json_conversions)
        if out is not None:
            return out

        mask = self.mask_as_bool(valid_when=True, nplike=self.nplike)[: self._length]
        out = self._content._getitem_range(slice(0, self._length))._to_list(
            behavior, json_conversions
        )

        for i, isvalid in enumerate(mask):
            if not isvalid:
                out[i] = None

        return out

    def _to_nplike(self, nplike):
        content = self._content._to_nplike(nplike)
        mask = self._mask._to_nplike(nplike)
        return BitMaskedArray(
            mask,
            content,
            valid_when=self._valid_when,
            length=len(self),
            lsb_order=self._lsb_order,
            identifier=self._identifier,
            parameters=self._parameters,
            nplike=nplike,
        )

    def _layout_equal(self, other, index_dtype=True, numpyarray=True):
        return (
            self.valid_when == other.valid_when
            and self.lsb_order == other.lsb_order
            and self.mask.layout_equal(other.mask, index_dtype, numpyarray)
            and self.content.layout_equal(other.content, index_dtype, numpyarray)
        )
