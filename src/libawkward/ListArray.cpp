// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>
#include <type_traits>

#include "awkward/cpu-kernels/identity.h"
#include "awkward/cpu-kernels/getitem.h"

#include "awkward/Slice.h"

#include "awkward/ListArray.h"

using namespace awkward;

template <typename T>
void ListArrayOf<T>::setid() {
  throw std::runtime_error("FIXME");
}

template <typename T>
void ListArrayOf<T>::setid(const std::shared_ptr<Identity> id) {
  throw std::runtime_error("FIXME");
}

template <typename T>
const std::string ListArrayOf<T>::tostring_part(const std::string indent, const std::string pre, const std::string post) const {
  std::stringstream out;
  std::string name = "Unrecognized ListArray";
  if (std::is_same<T, int32_t>::value) {
    name = "ListArray32";
  }
  else if (std::is_same<T, int64_t>::value) {
    name = "ListArray64";
  }
  out << indent << pre << "<" << name << ">\n";
  if (id_.get() != nullptr) {
    out << id_.get()->tostring_part(indent + std::string("    "), "", "\n");
  }
  out << starts_.tostring_part(indent + std::string("    "), "<starts>", "</starts>\n");
  out << stops_.tostring_part(indent + std::string("    "), "<stops>", "</stops>\n");
  out << content_.get()->tostring_part(indent + std::string("    "), "<content>", "</content>\n");
  out << indent << "</" << name << ">" << post;
  return out.str();
}

template <typename T>
int64_t ListArrayOf<T>::length() const {
  return starts_.length();
}

template <typename T>
const std::shared_ptr<Content> ListArrayOf<T>::shallow_copy() const {
  return std::shared_ptr<Content>(new ListArrayOf<T>(id_, starts_, stops_, content_));
}

template <typename T>
const std::shared_ptr<Content> ListArrayOf<T>::getitem_at(int64_t at) const {
  int64_t regular_at = at;
  if (regular_at < 0) {
    regular_at += starts_.length();
  }
  if (regular_at < 0  ||  regular_at >= starts_.length()) {
    throw std::invalid_argument("index out of range");
  }
  if (regular_at >= stops_.length()) {
    throw std::invalid_argument("len(stops) < len(starts) in ListArray");
  }
  return content_.get()->getitem_range(starts_.getitem_at(regular_at), stops_.getitem_at(regular_at));
}

template <typename T>
const std::shared_ptr<Content> ListArrayOf<T>::getitem_range(int64_t start, int64_t stop) const {
  int64_t regular_start = start;
  int64_t regular_stop = stop;
  awkward_regularize_rangeslice(regular_start, regular_stop, true, start != Slice::none(), stop != Slice::none(), starts_.length());
  if (regular_stop > stops_.length()) {
    throw std::invalid_argument("len(stops) < len(starts) in ListArray");
  }

  std::shared_ptr<Identity> id(nullptr);
  if (id_.get() != nullptr) {
    if (regular_stop > id_.get()->length()) {
      throw std::invalid_argument("index out of range for identity");
    }
    id = id_.get()->getitem_range(regular_start, regular_stop);
  }

  return std::shared_ptr<Content>(new ListArrayOf<T>(id, starts_.getitem_range(regular_start, regular_stop), stops_.getitem_range(regular_start, regular_stop), content_));
}

template <typename T>
const std::shared_ptr<Content> ListArrayOf<T>::getitem(const Slice& where) const {
  throw std::runtime_error("FIXME");
}

template <typename T>
const std::pair<int64_t, int64_t> ListArrayOf<T>::minmax_depth() const {
  std::pair<int64_t, int64_t> content_depth = content_.get()->minmax_depth();
  return std::pair<int64_t, int64_t>(content_depth.first + 1, content_depth.second + 1);
}

namespace awkward {
  template class ListArrayOf<int32_t>;
  template class ListArrayOf<int64_t>;
}
