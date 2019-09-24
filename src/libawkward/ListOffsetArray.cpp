// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>
#include <type_traits>

#include "awkward/cpu-kernels/identity.h"

#include "awkward/ListOffsetArray.h"

using namespace awkward;

template <typename T>
void ListOffsetArrayOf<T>::setid(const std::shared_ptr<Identity> id) {
  std::shared_ptr<Identity> theid = id;
  if (theid.get() == nullptr) {
    content_.get()->setid(theid);
  }
  else {
    if (length() != id.get()->length()) {
      throw std::invalid_argument("content and its id must have the same length");
    }
    if (std::is_same<T, int64_t>::value) {
      theid = theid.get()->to64();
    }
    if (Identity64* rawid = dynamic_cast<Identity64*>(theid.get())) {
      Identity64* rawsubid = new Identity64(Identity::newref(), rawid->fieldloc(), rawid->width() + 1, content_.get()->length());
      std::shared_ptr<Identity> newsubid(rawsubid);
      if (std::is_same<T, int32_t>::value) {
        awkward_identity64_from_listoffsets32(rawsubid->ptr().get(), rawid->ptr().get(), content_.get()->length(), reinterpret_cast<int32_t*>(offsets_.ptr().get()), rawid->width(), length());
      }
      else {
        awkward_identity64_from_listoffsets64(rawsubid->ptr().get(), rawid->ptr().get(), content_.get()->length(), reinterpret_cast<int64_t*>(offsets_.ptr().get()), rawid->width(), length());
      }
      content_.get()->setid(newsubid);
    }
    else if (Identity32* rawid = dynamic_cast<Identity32*>(theid.get())) {
      Identity32* rawsubid = new Identity32(Identity::newref(), rawid->fieldloc(), rawid->width() + 1, content_.get()->length());
      std::shared_ptr<Identity> newsubid(rawsubid);
      awkward_identity32_from_listoffsets32(rawsubid->ptr().get(), rawid->ptr().get(), content_.get()->length(), reinterpret_cast<int32_t*>(offsets_.ptr().get()), rawid->width(), length());
      content_.get()->setid(newsubid);
    }
  }
  id_ = theid;
}
template <typename T>
void ListOffsetArrayOf<T>::setid() {
  if (length() <= kMaxInt32) {
    Identity32* rawid = new Identity32(Identity::newref(), Identity::FieldLoc(), 1, length());
    std::shared_ptr<Identity> newid(rawid);
    awkward_new_identity32(rawid->ptr().get(), length());
    setid(newid);
  }
  else {
    Identity64* rawid = new Identity64(Identity::newref(), Identity::FieldLoc(), 1, length());
    std::shared_ptr<Identity> newid(rawid);
    awkward_new_identity64(rawid->ptr().get(), length());
    setid(newid);
  }
}

template <typename T>
const std::string ListOffsetArrayOf<T>::tostring_part(const std::string indent, const std::string pre, const std::string post) const {
  std::stringstream out;
  std::string name = "Unrecognized ListOffsetArray";
  if (std::is_same<T, int32_t>::value) {
    name = "ListOffsetArray32";
  }
  else if (std::is_same<T, int64_t>::value) {
    name = "ListOffsetArray64";
  }
  out << indent << pre << "<" << name << ">\n";
  if (id_.get() != nullptr) {
    out << id_.get()->tostring_part(indent + std::string("    "), "", "\n");
  }
  out << offsets_.tostring_part(indent + std::string("    "), "<offsets>", "</offsets>\n");
  out << content_.get()->tostring_part(indent + std::string("    "), "<content>", "</content>\n");
  out << indent << "</" << name << ">" << post;
  return out.str();
}

template <typename T>
int64_t ListOffsetArrayOf<T>::length() const {
  return offsets_.length() - 1;
}

template <typename T>
const std::shared_ptr<Content> ListOffsetArrayOf<T>::shallow_copy() const {
  return std::shared_ptr<Content>(new ListOffsetArrayOf<T>(id_, offsets_, content_));
}

template <typename T>
const std::shared_ptr<Content> ListOffsetArrayOf<T>::getitem_at(int64_t at) const {
  int64_t start = (int64_t)offsets_.getitem_at(at);
  int64_t stop = (int64_t)offsets_.getitem_at(at + 1);
  return content_.get()->getitem_range(start, stop);
}

template <typename T>
const std::shared_ptr<Content> ListOffsetArrayOf<T>::getitem_range(int64_t start, int64_t stop) const {
  std::shared_ptr<Identity> id(nullptr);
  if (id_.get() != nullptr) {
    id = id_.get()->getitem_range(start, stop);
  }
  return std::shared_ptr<Content>(new ListOffsetArrayOf<T>(id, offsets_.getitem_range(start, stop + 1), content_));
}

template <typename T>
const std::shared_ptr<Content> ListOffsetArrayOf<T>::getitem(const Slice& where) const {
  throw std::runtime_error("FIXME");
}

template <typename T>
const std::shared_ptr<Content> ListOffsetArrayOf<T>::getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& carry, const Index64& advanced) const {
  if (head.get() == nullptr) {
        throw std::runtime_error("ListOffsetArray[null]");
  }

  else if (SliceAt* at = dynamic_cast<SliceAt*>(head.get())) {
    throw std::runtime_error("ListOffsetArray[at]");
  }

  else if (SliceRange* range = dynamic_cast<SliceRange*>(head.get())) {
    throw std::runtime_error("ListOffsetArray[range]");
  }

  else if (SliceEllipsis* ellipsis = dynamic_cast<SliceEllipsis*>(head.get())) {
    throw std::runtime_error("ListOffsetArray[ellipsis]");
  }

  else if (SliceNewAxis* newaxis = dynamic_cast<SliceNewAxis*>(head.get())) {
    throw std::runtime_error("ListOffsetArray[newaxis]");
  }

  else if (SliceArray64* array = dynamic_cast<SliceArray64*>(head.get())) {
    throw std::runtime_error("ListOffsetArray[array]");
  }

  else {
    throw std::runtime_error("unrecognized slice item type");
  }
}

template <typename T>
const std::pair<int64_t, int64_t> ListOffsetArrayOf<T>::minmax_depth() const {
  std::pair<int64_t, int64_t> content_depth = content_.get()->minmax_depth();
  return std::pair<int64_t, int64_t>(content_depth.first + 1, content_depth.second + 1);
}

namespace awkward {
  template class ListOffsetArrayOf<int32_t>;
  template class ListOffsetArrayOf<int64_t>;
}
