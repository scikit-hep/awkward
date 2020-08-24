// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/partition/IrregularlyPartitionedArray.cpp", line)

#include <sstream>

#include "awkward/array/UnionArray.h"

#include "awkward/partition/IrregularlyPartitionedArray.h"

namespace awkward {
  IrregularlyPartitionedArray::IrregularlyPartitionedArray(
    const ContentPtrVec& partitions, const std::vector<int64_t> stops)
      : PartitionedArray(partitions)
      , stops_(stops) {
    if (partitions.size() != stops.size()) {
      throw std::invalid_argument(
        std::string("IrregularlyPartitionedArray stops must have the same length "
                    "as its partitions") + FILENAME(__LINE__));
    }
  }

  const std::vector<int64_t>
  IrregularlyPartitionedArray::stops() const {
    return stops_;
  }

  int64_t
  IrregularlyPartitionedArray::start(int64_t partitionid) const {
    if (partitionid == 0) {
      return 0;
    }
    else {
      return stops_[(size_t)(partitionid - 1)];
    }
  }

  int64_t
  IrregularlyPartitionedArray::stop(int64_t partitionid) const {
    return stops_[(size_t)partitionid];
  }

  void
  IrregularlyPartitionedArray::partitionid_index_at(int64_t at,
                                                    int64_t& partitionid,
                                                    int64_t& index) const {
    if (at < 0) {
      partitionid = -1;
      index = -1;
      return;
    }
    int64_t start = 0;
    for (int64_t i = 0;  i < numpartitions();  i++) {
      if (at < stops_[(size_t)i]) {
        partitionid = i;
        index = at - start;
        return;
      }
      start = stops_[(size_t)i];
    }
    partitionid = numpartitions();
    index = 0;
  }

  PartitionedArrayPtr
  IrregularlyPartitionedArray::repartition(
    const std::vector<int64_t>& stops) const {
    if (stops == stops_) {
      return shallow_copy();
    }
    if (stops.back() != stops_.back()) {
      throw std::invalid_argument(
        std::string("cannot repartition array of length ")
        + std::to_string(stops_.back()) + std::string(" to length ")
        + std::to_string(stops.back()) + FILENAME(__LINE__));
    }

    int64_t partitionid = 0;
    int64_t index = 0;
    ContentPtrVec partitions;
    for (uint64_t i = 0;  i < (uint64_t)stops.size();  i++) {
      ContentPtr dst(nullptr);
      int64_t length = (i == 0 ? stops[i]
                               : stops[i] - stops[i - 1]);

      while (dst.get() == nullptr  ||  dst.get()->length() < length) {
        ContentPtr piece(nullptr);
        ContentPtr src = partitions_[(size_t)partitionid];
        int64_t available = src.get()->length() - index;
        int64_t desired = (dst.get() == nullptr ? length
                                                : length - dst.get()->length());

        if (available <= desired) {
          piece = src.get()->getitem_range_nowrap(index, src.get()->length());
          partitionid++;
          index = 0;
        }
        else {
          piece = src.get()->getitem_range_nowrap(index, index + desired);
          index += desired;
        }

        if (dst.get() == nullptr) {
          dst = piece;
        }
        else {
          if (!dst.get()->mergeable(piece, false)) {
            dst = dst.get()->merge_as_union(piece);
          }
          else {
            dst = dst.get()->merge(piece);
          }
          if (UnionArray8_32* raw
                  = dynamic_cast<UnionArray8_32*>(dst.get())) {
            dst = raw->simplify_uniontype(false);
          }
          else if (UnionArray8_U32* raw
                       = dynamic_cast<UnionArray8_U32*>(dst.get())) {
            dst = raw->simplify_uniontype(false);
          }
          else if (UnionArray8_64* raw
                       = dynamic_cast<UnionArray8_64*>(dst.get())) {
            dst = raw->simplify_uniontype(false);
          }
        }
      }

      partitions.push_back(dst);
    }

    return std::make_shared<IrregularlyPartitionedArray>(partitions, stops);
  }

  const std::string
  IrregularlyPartitionedArray::classname() const {
    return "IrregularlyPartitionedArray";
  }

  const std::string
  IrregularlyPartitionedArray::tostring() const {
    std::stringstream out;
    out << "<" << classname() << ">\n";
    for (int64_t i = 0;  i < numpartitions();  i++) {
      out << "    <partition start=\"" << start(i) << "\" stop=\""
          << stop(i) << "\">\n";
      out << partition(i).get()->tostring_part("        ", "", "\n");
      out << "    </partition>\n";
    }
    out << "</" << classname() << ">";
    return out.str();
  }

  int64_t
  IrregularlyPartitionedArray::length() const {
    return stops_.back();
  }

  const PartitionedArrayPtr
  IrregularlyPartitionedArray::shallow_copy() const {
    return std::make_shared<IrregularlyPartitionedArray>(partitions_, stops_);
  }

  const PartitionedArrayPtr
  IrregularlyPartitionedArray::copy_to(kernel::lib ptr_lib) const {
    ContentPtrVec partitions;
    for (auto content : partitions_) {
      partitions.push_back(content.get()->copy_to(ptr_lib));
    }
    return std::make_shared<IrregularlyPartitionedArray>(partitions, stops_);
  }

}
