// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <cassert>
#include <atomic>
#include <iomanip>
#include <sstream>
#include <type_traits>
// #include <utility>

#include "awkward/Identity.h"

using namespace awkward;

std::atomic<Identity::Ref> numrefs{0};

Identity::Ref Identity::newref() {
  return numrefs++;
}

template <typename T>
const std::string IdentityOf<T>::tostring_part(const std::string indent, const std::string pre, const std::string post) const {
  std::stringstream out;
  std::string name = "Unrecognized Identity";
  if (std::is_same<T, int32_t>::value) {
    name = "Identity32";
  }
  else if (std::is_same<T, int64_t>::value) {
    name = "Identity64";
  }
  out << indent << pre << "<" << name << " ref=\"" << ref() << "\" fieldloc=\"[";
  for (size_t i = 0;  i < fieldloc().size();  i++) {
    if (i != 0) {
      out << " ";
    }
    out << "(" << fieldloc()[i].first << ", '" << fieldloc()[i].second << "')";
  }
  out << "]\" width=\"" << width() << "\" length=\"" << length() << "\" at=\"0x";
  out << std::hex << std::setw(12) << std::setfill('0') << reinterpret_cast<ssize_t>(ptr_.get()) << "\"/>" << post;
  return out.str();
}

template <typename T>
const std::string IdentityOf<T>::tostring() const {
  return tostring_part("", "", "");
}

template <typename T>
const std::shared_ptr<Identity> IdentityOf<T>::slice(int64_t start, int64_t stop) const {
  return std::shared_ptr<Identity>(new IdentityOf<T>(ref(), fieldloc(), offset() + width()*start*(start != stop), width(), (stop - start), ptr_));
}

template <typename T>
const std::shared_ptr<Identity> IdentityOf<T>::shallow_copy() const {
  return std::shared_ptr<Identity>(new IdentityOf<T>(ref(), fieldloc(), offset(), width(), length(), ptr_));
}

template <typename T>
const std::vector<T> IdentityOf<T>::get(int64_t at) const {
  std::vector<T> out;
  for (size_t i = (size_t)(offset() + at);  i < (size_t)(offset() + at + width());  i++) {
    out.push_back(ptr_.get()[i]);
  }
  return out;
}

namespace awkward {
  template class IdentityOf<int32_t>;
  template class IdentityOf<int64_t>;
}
