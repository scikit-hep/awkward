// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>

#include "awkward/partition/IrregularlyPartitionedArray.h"

namespace awkward {
  IrregularlyPartitionedArray::IrregularlyPartitionedArray(
    const ContentPtrVec& partitions, const std::vector<int64_t> stops)
      : PartitionedArray(partitions)
      , stops_(stops) {
    if (partitions.size() != stops.size()) {
      throw std::invalid_argument(
        "IrregularlyPartitionedArray stops must have the same length "
        "as its partitions");
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
    return stops_[(size_t)((int64_t)stops_.size() - 1)];
  }

  const PartitionedArrayPtr
  IrregularlyPartitionedArray::shallow_copy() const {
    return std::make_shared<IrregularlyPartitionedArray>(partitions_, stops_);
  }
}
