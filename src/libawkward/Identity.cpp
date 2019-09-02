// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Identity.h"

using namespace awkward;

std::atomic<Ref> numrefs{0};

Ref newref() {
  return numrefs++;
}

template <typename T>
const std::string IdentityOf<T>::repr(const std::string indent, const std::string pre, const std::string post) const {
  std::stringstream out;
  std::string name = "Unrecognized Identity";
  if (std::is_same<T, int32_t>::value) {
    name = "Identity32";
  }
  else if (std::is_same<T, int64_t>::value) {
    name = "Identity64";
  }
  out << indent << pre << "<" << name << " ref=\"" << ref_ << "\" fieldloc=\"[";
  for (int64_t i = 0;  i < fieldloc_.size();  i++) {
    if (i != 0) {
      out << " ";
    }
    out << "(" << fieldloc_[i].first << ", '" << fieldloc_[i].second << "')";
  }
  out << "]\" width=\"" << width() << "\" length=\"" << length_ << "\" at=\"0x";
  out << std::hex << std::setw(12) << std::setfill('0') << reinterpret_cast<ssize_t>(ptr_.get()) << "\"/>" << post;
  return out.str();
}

template <typename T>
const std::shared_ptr<Identity> IdentityOf<T>::slice(int64_t start, int64_t stop) const {
  return std::shared_ptr<Identity>(new IdentityOf<T>(ref_, fieldloc_, ptr_, offset_ + width_*start*(start != stop), width_, (stop - start)));
}

namespace awkward {
  template class IdentityOf<int32_t> Identity32;
  template class IdentityOf<int64_t> Identity64;
}
