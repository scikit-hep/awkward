# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

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
        return self.from_ext(self._ext.repartition(*args, **kwargs))

    def __getitem__(self, where):
        if isinstance(where, int):
            return self._ext.getitem_at(where)

        elif isinstance(where, slice) and (where.step is None or
                                           where.step == 1):
            return self._ext.getitem_range(where.start, where.stop)

        elif (isinstance(where, str) or
              (awkward1._util.py27 and isinstance(where, unicode))):
            raise NotImplementedError

        elif (instance(where, Iterable) and
              all((isinstance(x, str) or
                   (awkward._util.py27 and isinstance(x, unicode)))
                  for x in where)):
            raise NotImplementedError

        else:
            if not isinstance(where, tuple):
                where = (where,)
            raise NotImplementedError

class IrregularlyPartitionedArray(PartitionedArray):
    def __init__(self, partitions):
        self._ext = awkward1._ext.IrregularlyPartitionedArray(partitions)
