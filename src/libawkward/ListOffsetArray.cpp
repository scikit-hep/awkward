// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/identity.h"
#include "awkward/ListOffsetArray.h"

using namespace awkward;

template <typename T>
void ListOffsetArrayOf<T>::setid() {
  std::shared_ptr<Identity> newid(new Identity(Identity::newref(), FieldLocation(), 1, length()));
  Error err = awkward_identity_new32(length(), newid.get()->ptr().get());
  HANDLE_ERROR(err);
  setid(newid);
}

template <typename T>
void ListOffsetArrayOf<T>::setid(const std::shared_ptr<Identity> id) {
  if (id.get() == nullptr) {
    content_.get()->setid(id);
  }
  else {
    std::shared_ptr<Identity> newid(new Identity32(Identity::newref(), id.get()->fieldloc(), id.get()->width() + 1, content_.get()->length()));
    Error err = awkward_identity_from_listfoffsets32(length(), id.get()->width(), offsets_.ptr().get(), id.get()->ptr().get(), content_.get()->length(), newid.get()->ptr().get());
    HANDLE_ERROR(err);
    content_.get()->setid(newid);
  }
  id_ = id;
}

template <typename T>
const std::string ListOffsetArrayOf<T>::repr(const std::string indent, const std::string pre, const std::string post) const {
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
    out << id_.get()->repr(indent + std::string("    "), "", "\n");
  }
  out << offsets_.repr(indent + std::string("    "), "<offsets>", "</offsets>\n");
  out << content_.get()->repr(indent + std::string("    "), "<content>", "</content>\n");
  out << indent << "</" << name << ">" << post;
  return out.str();
}

template <typename T>
int64_t ListOffsetArrayOf<T>::length() const {
  return offsets_.length() - 1;
}

template <typename T>
std::shared_ptr<Content> ListOffsetArrayOf<T>::shallow_copy() const {
  return std::shared_ptr<Content>(new ListOffsetArrayOf<T>(id_, offsets_, content_));
}

template <typename T>
std::shared_ptr<Content> ListOffsetArrayOf<T>::get(int64_t at) const {
  int64_t start = (int64_t)offsets_.get(at);
  int64_t stop = (int64_t)offsets_.get(at + 1);
  return content_.get()->slice(start, stop);
}

template <typename T>
std::shared_ptr<Content> ListOffsetArrayOf<T>::slice(int64_t start, int64_t stop) const {
  std::shared_ptr<Identity> id;
  if (id_.get() != nullptr) {
    id = id_.get()->slice(start, stop);
  }
  return std::shared_ptr<Content>(new ListOffsetArrayOf<T>(id, offsets_.slice(start, stop + 1), content_));
}

namespace awkward {
  template class ListOffsetArrayOf<int32_t> ListOffsetArray32;
  template class ListOffsetArrayOf<int64_t> ListOffsetArray64;
}
