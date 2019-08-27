// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Identity.h"

using namespace awkward;

std::atomic<RefType> numrefs{0};

RefType Identity::newref() {
  return numrefs++;
}

IndexType Identity::keydepth(IndexType chunkdepth, IndexType indexdepth) {
  return (sizeof(ChunkOffsetType)/sizeof(IndexType))*chunkdepth + indexdepth;
}

const std::string Identity::repr(const std::string indent, const std::string pre, const std::string post) const {
  std::stringstream out;
  out << indent << pre << "<Identity ref=\"" << ref_ << "\" fieldloc=\"[";
  for (int i = 0;  i < fieldloc_.size();  i++) {
    if (i != 0) {
      out << " ";
    }
    out << "(" << fieldloc_[i].first << ", '" << fieldloc_[i].second << "')";
  }
  out << "]\" keydepth=\"" << keydepth() << "\" length=\"" << length_ << "\" at=\"0x";
  out << std::hex << std::setw(12) << std::setfill('0') << reinterpret_cast<ssize_t>(ptr_.get()) << "\"/>" << post;
  return out.str();
}
