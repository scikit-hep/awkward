// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/partition/IrregularlyPartitionedArray.h"

#include "awkward/partition/PartitionedArray.h"

namespace awkward {
  PartitionedArray::PartitionedArray(const ContentPtrVec& partitions)
      : partitions_(partitions) {
    if (partitions_.empty()) {
      throw std::invalid_argument(
        "PartitionedArray must have at least one partition");
    }
  }

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
    return partitions_[(size_t)partitionindex];
  }

  const std::string
  PartitionedArray::tojson(bool pretty, int64_t maxdecimals) const {
    if (pretty) {
      ToJsonPrettyString builder(maxdecimals);
      builder.beginlist();
      for (auto p : partitions_) {
        p.get()->tojson_part(builder, false);
      }
      builder.endlist();
      return builder.tostring();
    }
    else {
      ToJsonString builder(maxdecimals);
      builder.beginlist();
      for (auto p : partitions_) {
        p.get()->tojson_part(builder, false);
      }
      builder.endlist();
      return builder.tostring();
    }
  }

  void
  PartitionedArray::tojson(FILE* destination,
                           bool pretty,
                           int64_t maxdecimals,
                           int64_t buffersize) const {
    if (pretty) {
      ToJsonPrettyFile builder(destination, maxdecimals, buffersize);
      builder.beginlist();
      for (auto p : partitions_) {
        p.get()->tojson_part(builder, false);
      }
      builder.endlist();
    }
    else {
      ToJsonFile builder(destination, maxdecimals, buffersize);
      builder.beginlist();
      for (auto p : partitions_) {
        p.get()->tojson_part(builder, false);
      }
      builder.endlist();
    }
  }

  const ContentPtr
  PartitionedArray::getitem_at(int64_t at) const {
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += length();
    }
    if (!(0 <= regular_at  &&  regular_at < length())) {
      util::handle_error(
        failure("index out of range", kSliceNone, at),
        classname(),
        nullptr);
    }
    return getitem_at_nowrap(regular_at);
  }

  const ContentPtr
  PartitionedArray::getitem_at_nowrap(int64_t at) const {
    int64_t partitionid;
    int64_t index;
    partitionid_index_at(at, partitionid, index);
    return partitions_[(size_t)partitionid].get()->getitem_at_nowrap(index);
  }

  const PartitionedArrayPtr
  PartitionedArray::getitem_range(int64_t start, int64_t stop) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    awkward_regularize_rangeslice(&regular_start, &regular_stop,
      true, start != Slice::none(), stop != Slice::none(), length());
    if (regular_stop > length()) {
      util::handle_error(
        failure("len(stops) < len(starts)", kSliceNone, kSliceNone),
        classname(),
        nullptr);
    }
    return getitem_range_nowrap(regular_start, regular_stop);
  }

  const PartitionedArrayPtr
  PartitionedArray::getitem_range_nowrap(int64_t start, int64_t stop) const {
    int64_t partitionid_first;
    int64_t index_start;
    partitionid_index_at(start, partitionid_first, index_start);
    int64_t partitionid_last;
    int64_t index_stop;
    partitionid_index_at(stop, partitionid_last, index_stop);
    if (index_stop == 0) {
      partitionid_last--;
      if (partitionid_last >= 0) {
        index_stop = partitions_[(size_t)partitionid_last].get()->length();
      }
    }
    ContentPtrVec partitions;
    std::vector<int64_t> stops;
    int64_t total_length = 0;
    for (int64_t partitionid = partitionid_first;
         partitionid <= partitionid_last;
         partitionid++) {
      ContentPtr p = partitions_[(size_t)partitionid];
      if (partitionid == partitionid_first  &&
          partitionid == partitionid_last) {
        p = p.get()->getitem_range_nowrap(index_start, index_stop);
      }
      else if (partitionid == partitionid_first) {
        p = p.get()->getitem_range_nowrap(index_start, p.get()->length());
      }
      else if (partitionid == partitionid_last) {
        p = p.get()->getitem_range_nowrap(0, index_stop);
      }
      total_length += p.get()->length();
      partitions.push_back(p);
      stops.push_back(total_length);
    }
    if (partitions.empty()) {
      partitions.push_back(partitions_[0].get()->getitem_nothing());
      stops.push_back(0);
    }
    return std::make_shared<IrregularlyPartitionedArray>(partitions, stops);
  }
}
