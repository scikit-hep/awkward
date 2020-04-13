// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/IrregularlyPartitionedArray.h"

#include "awkward/PartitionedArray.h"

namespace awkward {
  PartitionedArray::PartitionedArray(const ContentPtrVec& partitions)
      : partitions_(partitions) { }

  const ContentPtrVec
  PartitionedArray::partitions() const {
    return partitions_;
  }

  int64_t
  PartitionedArray::numpartitions() const {
    return (int64_t)partitions_.size();
  }

  const ContentPtr
  PartitionedArray::partition(int64_t partitionindex) const {
    if (!(0 <= partitionindex  &&  partitionindex < numpartitions())) {
      throw std::invalid_argument("partitionindex out of bounds");
    }
    return partitions_[partitionindex];
  }

  const ContentPtr
  PartitionedArray::getitem_at_nowrap(int64_t at) const {
    partitionid_index_at_nowrap(at)
  }
}
