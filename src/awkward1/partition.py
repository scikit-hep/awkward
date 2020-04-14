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

class PartitionedArray(object):
    @classmethod
    def from_ext(cls, obj):
        if isinstance(obj, awkward1._ext.IrregularlyPartitionedArray):
            subcls = IrregularlyPartitionedArray
        else:
            raise AssertionError("unrecognized PartitionedArray: {0}".format(
                                     str(type(obj))))

        out = subcls.__new__(subcls)
        out._ext = obj
        return out

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

    def tojson(self, *args, **kwargs):
        return self._ext.tojson(*args, **kwargs)

    def __len__(self):
        return len(self._ext)

    def __iter__(self):
        for partition in self.partitions:
            for x in partition:
                yield x

    def __array__(self):
        raise NotImplementedError

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
        return PartitionedArray.from_ext(self._ext.repartition(*args, **kwargs))

    def __getitem__(self, where):
        import awkward1.operations.convert
        import awkward1.operations.describe
        from awkward1.types import PrimitiveType, OptionType

        if (not isinstance(where, bool) and
            isinstance(where, (numbers.Integral, numpy.integer))):
            return self._ext.getitem_at(where)

        elif isinstance(where, slice):
            return self._ext.getitem_range(where.start, where.stop, where.step)

        elif (isinstance(where, str) or
              (awkward1._util.py27 and isinstance(where, unicode))):
            return self.replace_partitions([x[where] for x in self.partitions])

        elif (isinstance(where, Iterable) and
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

    def replace_partitions(self, partitions):
        return awkward1._ext.IrregularlyPartitionedArray(partitions, self.stops)
