// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/partition/PartitionedArray.cpp", line)
#define FILENAME_C(line) FILENAME_FOR_EXCEPTIONS_C("src/libawkward/partition/PartitionedArray.cpp", line)

#include "awkward/partition/IrregularlyPartitionedArray.h"

#include "awkward/partition/PartitionedArray.h"

namespace awkward {
  PartitionedArray::PartitionedArray(const ContentPtrVec& partitions)
      : partitions_(partitions) {
    if (partitions_.empty()) {
      throw std::invalid_argument(
        std::string("PartitionedArray must have at least one partition")
        + FILENAME(__LINE__));
    }
  }

  PartitionedArray::~PartitionedArray() = default;

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
      throw std::invalid_argument(
        std::string("partitionindex out of bounds") + FILENAME(__LINE__));
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
        failure("index out of range", kSliceNone, at, FILENAME_C(__LINE__)),
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
  PartitionedArray::getitem_range(int64_t start,
                                  int64_t stop,
                                  int64_t step) const {
    int64_t regular_start = start;
    int64_t regular_stop = stop;
    int64_t regular_step = step;
    if (regular_step == Slice::none()) {
      regular_step = 1;
    }
    kernel::regularize_rangeslice(&regular_start, &regular_stop,
      step > 0, start != Slice::none(), stop != Slice::none(), length());
    return getitem_range_nowrap(regular_start, regular_stop, regular_step);
  }

  const PartitionedArrayPtr
  PartitionedArray::getitem_range_nowrap(int64_t start,
                                         int64_t stop,
                                         int64_t step) const {
    int64_t partitionid_first;
    int64_t index_start;
    partitionid_index_at(start, partitionid_first, index_start);
    int64_t partitionid_last;
    int64_t index_stop;
    partitionid_index_at(stop, partitionid_last, index_stop);

    ContentPtrVec partitions;
    std::vector<int64_t> stops;
    int64_t total_length = 0;
    int64_t offset = 0;

    if (step > 0) {
      for (int64_t partitionid = partitionid_first;
           partitionid < numpartitions()  &&  partitionid <= partitionid_last;
           partitionid++) {
        ContentPtr p = partitions_[(size_t)partitionid];
        int64_t plen = p.get()->length();

        if (partitionid == partitionid_first  &&
            partitionid == partitionid_last) {
          if (step == 1) {
            p = p.get()->getitem_range_nowrap(index_start, index_stop);
          }
          else {
            Slice slice;
            slice.append(SliceRange(index_start, index_stop, step));
            slice.become_sealed();
            p = p.get()->getitem(slice);
          }
        }
        else if (partitionid == partitionid_first) {
          if (step == 1) {
            p = p.get()->getitem_range_nowrap(index_start, plen);
          }
          else {
            Slice slice;
            slice.append(SliceRange(index_start, plen, step));
            slice.become_sealed();
            p = p.get()->getitem(slice);
            offset = ((index_start - plen) % step + step) % step;
          }
        }
        else if (partitionid == partitionid_last) {
          if (step == 1) {
            p = p.get()->getitem_range_nowrap(0, index_stop);
          }
          else {
            Slice slice;
            slice.append(SliceRange(offset, index_stop, step));
            slice.become_sealed();
            p = p.get()->getitem(slice);
          }
        }
        else if (step != 1) {
          Slice slice;
          slice.append(SliceRange(offset, plen, step));
          slice.become_sealed();
          p = p.get()->getitem(slice);
          offset = ((offset - plen) % step + step) % step;
        }

        total_length += p.get()->length();
        if (p.get()->length() > 0) {
          partitions.push_back(p);
          stops.push_back(total_length);
        }
      }
    }

    else if (step < 0) {
      for (int64_t partitionid = partitionid_first;
           partitionid >= 0  &&  partitionid >= partitionid_last;
           partitionid--) {
        ContentPtr p = partitions_[(size_t)partitionid];
        int64_t plen = p.get()->length();

        int64_t a;
        int64_t b;
        if (partitionid == partitionid_first  &&
            partitionid == partitionid_last) {
          a = index_start;
          b = index_stop;
        }
        else if (partitionid == partitionid_first) {
          a = index_start;
          b = -plen - 1;
          offset = (((-1 - index_start) % -step + -step) % -step);
        }
        else if (partitionid == partitionid_last) {
          a = plen - 1 - offset;
          b = index_stop;
        }
        else {
          a = plen - 1 - offset;
          b = -plen - 1;
          offset = (((-1 - (plen - 1 - offset)) % -step + -step) % -step);
        }
        // Avoid Python-like negative index handling of -1 by setting them to
        // a sufficiently negative value to mean "all the way to the edge."
        if (a < 0) {
          a = -plen - 1;
        }
        if (b < 0) {
          b = -plen - 1;
        }

        Slice slice;
        slice.append(SliceRange(a, b, step));
        slice.become_sealed();
        p = p.get()->getitem(slice);

        total_length += p.get()->length();
        if (p.get()->length() > 0) {
          partitions.push_back(p);
          stops.push_back(total_length);
        }
      }
    }

    else {
      throw std::invalid_argument(
        std::string("slice step must not be zero") + FILENAME(__LINE__));
    }

    if (partitions.empty()) {
      partitions.push_back(partitions_[0].get()->getitem_nothing());
      stops.push_back(0);
    }
    return std::make_shared<IrregularlyPartitionedArray>(partitions, stops);
  }
}
