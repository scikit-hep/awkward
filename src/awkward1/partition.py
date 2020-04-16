# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import numbers
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numpy

import awkward1._ext
import awkward1._util

def single(obj):
    if isinstance(obj, awkward1.layout.Content):
        return IrregularlyPartitionedArray([obj])
    else:
        return obj

def first(obj):
    if isinstance(obj, PartitionedArray):
        return obj.partition(0)
    else:
        return obj

def every(obj):
    if isinstance(obj, PartitionedArray):
        return obj.partitions
    else:
        return [obj]

def partition_as(sample, arrays):
    stops = sample.stops
    if isinstance(arrays, dict):
        out = {}
        for n, x in arrays.items():
            if isinstance(x, PartitionedArray):
                out[n] = x.repartition(stops)
            elif isinstance(x, awkward1.layout.Content):
                out[n] = IrregularlyPartitionedArray.toPartitioned(x, stops)
            else:
                out[n] = x
        return out
    else:
        out = []
        for x in arrays:
            if isinstance(x, PartitionedArray):
                out.append(x.repartition(stops))
            elif isinstance(x, awkward1.layout.Content):
                out.append(IrregularlyPartitionedArray.toPartitioned(x, stops))
            else:
                out.append(x)
        return out

def iterate(numpartitions, arrays):
    if isinstance(arrays, dict):
        for partitionid in range(numpartitions):
            out = {}
            for n, x in arrays.items():
                if isinstance(x, PartitionedArray):
                    out[n] = x.partition(partitionid)
                else:
                    out[n] = x
            yield out
    else:
        for partitionid in range(numpartitions):
            out = []
            for x in arrays:
                if isinstance(x, PartitionedArray):
                    out.append(x.partition(partitionid))
                else:
                    out.append(x)
            yield out

def apply(function, array):
    return IrregularlyPartitionedArray([function(x) for x in array.partitions])

class PartitionedArray(object):
    @classmethod
    def from_ext(cls, obj):
        if isinstance(obj, awkward1._ext.IrregularlyPartitionedArray):
            subcls = IrregularlyPartitionedArray
            out = subcls.__new__(subcls)
            out._ext = obj
            return out
        else:
            return obj

    def __repr__(self):
        return repr(self._ext)

    @property
    def partitions(self):
        return self._ext.partitions

    @property
    def numpartitions(self):
        return self._ext.numpartitions

    def partition(self, partitionid):
        return self._ext.partition(partitionid)

    def start(self, partitionid):
        return self._ext.start(partitionid)

    def stop(self, partitionid):
        return self._ext.stop(partitionid)

    def partitionid_index_at(self, at):
        return self._ext.partitionid_index_at(at)

    def type(self, typestrs):
        ts = []
        for x in self.partitions:
            t = x.type(typestrs)
            if not isinstance(t, awkward1.types.UnknownType):
                option = None
                for i, ti in enumerate(ts):
                    if t == ti:
                        break
                    elif t == awkward1.types.OptionType(ti):
                        option = t
                        break
                    elif awkward1.types.OptionType(t) == ti:
                        option = ti
                        break
                else:
                    ts.append(t)

                if option is not None:
                    del ts[i]
                    ts.append(option)

        if len(ts) == 0:
            return awkward1.types.UnknownType()
        elif len(ts) == 1:
            return ts[0]
        else:
            return awkward1.types.UnionType(ts)

    @property
    def parameters(self):
        return first(self).parameters

    def parameter(self, *args, **kwargs):
        return first(self).parameter(*args, **kwargs)

    def purelist_parameter(self, *args, **kwargs):
        return first(self).purelist_parameter(*args, **kwargs)

    def tojson(self, *args, **kwargs):
        return self._ext.tojson(*args, **kwargs)

    @property
    def nbytes(self):
        return sum(x.nbytes for x in self.partitions)

    def deep_copy(self, *args, **kwargs):
        out = type(self).__new__(type(self))
        out.__dict__.update(self.__dict__)
        out.partitions = [x.deep_copy(*args, **kwargs) for x in out.partitions]
        return out

    @property
    def numfields(self):
        return first(self).numfields

    def fieldindex(self, *args, **kwargs):
        return first(self).fieldindex(*args, **kwargs)

    def key(self, *args, **kwargs):
        return first(self).key(*args, **kwargs)

    def haskey(self, *args, **kwargs):
        return first(self).haskey(*args, **kwargs)

    def keys(self, *args, **kwargs):
        return first(self).keys(*args, **kwargs)

    @property
    def purelist_isregular(self):
        return first(self).purelist_isregular

    @property
    def purelist_depth(self):
        return first(self).purelist_depth

    def getitem_nothing(self, *args, **kwargs):
        return first(self).getitem_nothing(*args, **kwargs)

    def validityerror(self, *args, **kwargs):
        for x in self.partitions:
            out = x.validityerror(*args, **kwargs)
            if out is not None:
                return out
        return None

    def flatten(self, *args, **kwargs):
        return apply(lambda x: x.flatten(*args, **kwargs), self)

    def argmin(self, axis, mask, keepdims):
        branch, depth = first(self).branch_depth
        negaxis = -axis
        if not branch and negaxis <= 0:
            negaxis += depth

        if not branch and negaxis == depth:
            return self.toContent().argmin(axis, mask, keepdims)
        else:
            return self.replace_partitions([x.argmin(axis, mask, keepdims)
                                              for x in self.partitions])

    def argmax(self, axis, mask, keepdims):
        branch, depth = first(self).branch_depth
        negaxis = -axis
        if not branch and negaxis <= 0:
            negaxis += depth

        if not branch and negaxis == depth:
            return self.toContent().argmax(axis, mask, keepdims)
        else:
            return self.replace_partitions([x.argmax(axis, mask, keepdims)
                                              for x in self.partitions])

    def localindex(self, axis):
        if first(self).axis_wrap_if_negative(axis) == 0:
            start = 0
            output = []
            for x in self.partitions:
                output.append(awkward1.layout.NumpyArray(
                    numpy.arange(start, start + len(x), dtype=numpy.int64)))
                start += len(x)
            return self.replace_partitions(output)

        else:
            return self.replace_partitions([x.localindex(axis)
                                              for x in self.partitions])

    def combinations(self, n, replacement, keys, parameters, axis):
        if first(self).axis_wrap_if_negative(axis) == 0:
            return self.toContent().combinations(n,
                                                 replacement,
                                                 keys,
                                                 parameters,
                                                 axis)
        else:
            return self.replace_partitions([x.combinations(n,
                                                           replacement,
                                                           keys,
                                                           parameters,
                                                           axis)
                                              for x in self.partitions])

    def __len__(self):
        return len(self._ext)

    def __iter__(self):
        for partition in self.partitions:
            for x in partition:
                yield x

    def __array__(self):
        tocat = []
        for x in self.partitions:
            y = awkward1.operations.convert.to_numpy(x)
            if len(y) > 0:
                tocat.append(y)

        if len(tocat) > 0:
            if any(isinstance(x, numpy.ma.MaskedArray) for x in tocat):
                return numpy.ma.concatenate(tocat)
            else:
                return numpy.concatenate(tocat)
        else:
            return y

    def toContent(self):
        contents = self._ext.partitions
        out = contents[0]
        for x in contents[1:]:
            if not out.mergeable(x, mergebool=False):
                out = out.merge_as_union(x)
            else:
                out = out.merge(x)
            if isinstance(out, awkward1._util.uniontypes):
                out = out.simplify(mergebool=mergebool)
        return out

    def repartition(self, *args, **kwargs):
        return PartitionedArray.from_ext(
                   self._ext.repartition(*args, **kwargs))

    def __getitem__(self, where):
        import awkward1.operations.convert
        import awkward1.operations.describe
        from awkward1.types import PrimitiveType, OptionType

        if (not isinstance(where, bool) and
            isinstance(where, (numbers.Integral, numpy.integer))):
            return PartitionedArray.from_ext(self._ext.getitem_at(where))

        elif isinstance(where, slice):
            return PartitionedArray.from_ext(self._ext.getitem_range(
                       where.start, where.stop, where.step))

        elif (isinstance(where, str) or
              (awkward1._util.py27 and isinstance(where, unicode))):
            return self.replace_partitions([x[where] for x in self.partitions])

        elif isinstance(where, tuple) and len(where) == 0:
            return self

        elif (isinstance(where, Iterable) and
              len(where) > 0 and
              all((isinstance(x, str) or
                   (awkward1._util.py27 and isinstance(x, unicode)))
                  for x in where)):
            return self.replace_partitions([x[where] for x in self.partitions])

        else:
            if not isinstance(where, tuple):
                where = (where,)
            head, tail = where[0], where[1:]

            if (not isinstance(head, bool) and
                isinstance(head, (numbers.Integral, numpy.integer))):
                original_head = head
                if head < 0:
                    head += len(self)
                if not 0 <= head < len(self):
                    raise ValueError(
                        "{0} index out of range".format(type(self).__name__))
                partitionid, index = self._ext.partitionid_index_at(head)
                return self.partition(partitionid)[(index,) + tail]

            elif isinstance(head, slice):
                partitions = self._ext.getitem_range(head.start,
                                                     head.stop,
                                                     head.step).partitions
                return IrregularlyPartitionedArray([x[(slice(None),) + tail]
                                                      for x in partitions])

            elif ((head is Ellipsis) or
                  (isinstance(where, str) or
                   (awkward1._util.py27 and isinstance(where, unicode))) or
                  (isinstance(head, Iterable) and
                   all((isinstance(x, str) or
                        (awkward1._util.py27 and isinstance(x, unicode)))
                       for x in head))):
                return IrregularlyPartitionedArray([x[(head,) + tail]
                                                      for x in self.partitions])

            elif head is numpy.newaxis:
                return self.toContent()[(head,) + tail]

            else:
                layout = awkward1.operations.convert.to_layout(head,
                             allow_record=False, allow_other=False,
                             numpytype=(numpy.integer, numpy.bool_, numpy.bool))
                t = awkward1.operations.describe.type(layout)
                if t in (PrimitiveType("int8"),
                         PrimitiveType("uint8"),
                         PrimitiveType("int16"),
                         PrimitiveType("uint16"),
                         PrimitiveType("int32"),
                         PrimitiveType("uint32"),
                         PrimitiveType("int64"),
                         PrimitiveType("uint64"),
                         OptionType(PrimitiveType("int8")),
                         OptionType(PrimitiveType("uint8")),
                         OptionType(PrimitiveType("int16")),
                         OptionType(PrimitiveType("uint16")),
                         OptionType(PrimitiveType("int32")),
                         OptionType(PrimitiveType("uint32")),
                         OptionType(PrimitiveType("int64")),
                         OptionType(PrimitiveType("uint64"))):
                    if isinstance(layout, PartitionedArray):
                        layout = layout.toContent()
                    return self.toContent()[(layout,) + tail]
                else:
                    stops = self.stops
                    if isinstance(layout, PartitionedArray):
                        layout = layout.repartition(stops)
                    else:
                        layout = IrregularlyPartitionedArray.toPartitioned(
                                     layout, stops)
                    inparts = self.partitions
                    headparts = layout.partitions
                    outparts = []
                    for i in range(len(inparts)):
                        outparts.append(inparts[i][(headparts[i],) + tail])
                    return IrregularlyPartitionedArray(outparts, stops)

class IrregularlyPartitionedArray(PartitionedArray):
    @classmethod
    def toPartitioned(cls, layout, stops):
        partitions = []
        start = 0
        for stop in stops:
            partitions.append(layout[start:stop])
            start = stop
        return IrregularlyPartitionedArray(partitions, stops)

    def __init__(self, partitions, stops=None):
        if stops is None:
            self._ext = awkward1._ext.IrregularlyPartitionedArray(partitions)
        else:
            self._ext = awkward1._ext.IrregularlyPartitionedArray(partitions,
                                                                  stops)

    @property
    def stops(self):
        return self._ext.stops

    def validityerror(self):
        total_length = 0
        for x, stop in zip(self.partitions, self.stops):
            total_length += len(x)
            if total_length != stop:
                return ("IrregularlyPartitionedArray stops do not match "
                        "partition lengths")
            out = x.validityerror()
            if out is not None:
                return out
        return None

    def replace_partitions(self, partitions):
        return IrregularlyPartitionedArray(partitions, self.stops)
