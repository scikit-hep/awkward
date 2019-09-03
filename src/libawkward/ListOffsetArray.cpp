// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/identity.h"
#include "awkward/ListOffsetArray.h"

using namespace awkward;

template <typename T>
void ListOffsetArrayOf<T>::setid() {
  Identity32* rawid = new Identity32(Identity::newref(), Identity::FieldLoc(), 1, length());
  std::shared_ptr<Identity> newid(rawid);
  Error err = awkward_identity_new32(length(), rawid->ptr().get());
  HANDLE_ERROR(err);
  setid(newid);
}

template <typename T>
void ListOffsetArrayOf<T>::setid(const std::shared_ptr<Identity> id) {
  if (id.get() == nullptr) {
    content_.get()->setid(id);
  }
  else {
    Identity32* rawid32 = dynamic_cast<Identity32*>(id.get());
    Identity64* rawid64 = dynamic_cast<Identity64*>(id.get());
    if (rawid32  &&  std::is_same<T, int32_t>::value) {
      Identity32* rawsubid = new Identity32(Identity::newref(), rawid32->fieldloc(), rawid32->width() + 1, content_.get()->length());
      std::shared_ptr<Identity> newsubid(rawsubid);
      Error err = awkward_identity_from_listfoffsets32(length(), rawid32->width(), reinterpret_cast<int32_t*>(offsets_.ptr().get()), rawid32->ptr().get(), content_.get()->length(), rawsubid->ptr().get());
      HANDLE_ERROR(err);
      content_.get()->setid(newsubid);
    }
    else {
      throw std::runtime_error("unhandled Identity specialization case");
    }
  }
  id_ = id;
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
const std::shared_ptr<Content> ListOffsetArrayOf<T>::get(int64_t at) const {
  int64_t start = (int64_t)offsets_.get(at);
  int64_t stop = (int64_t)offsets_.get(at + 1);
  return content_.get()->slice(start, stop);
}

template <typename T>
const std::shared_ptr<Content> ListOffsetArrayOf<T>::slice(int64_t start, int64_t stop) const {
  std::shared_ptr<Identity> id(nullptr);
  if (id_.get() != nullptr) {
    id = id_.get()->slice(start, stop);
  }
  return std::shared_ptr<Content>(new ListOffsetArrayOf<T>(id, offsets_.slice(start, stop + 1), content_));
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
