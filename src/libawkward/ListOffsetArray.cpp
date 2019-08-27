// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/identity.h"
#include "awkward/ListOffsetArray.h"

using namespace awkward;

void ListOffsetArray::setid() {
  std::shared_ptr<Identity> newid(new Identity(Identity::newref(), FieldLocation(), 0, 1, length()));
  Error err = awkward_identity_new(length(), newid.get()->ptr().get());
  HANDLE_ERROR(err);
  setid(newid);
}

void ListOffsetArray::setid(const std::shared_ptr<Identity> id) {
  if (id.get() == nullptr) {
    content_.get()->setid(id);
  }
  else {
    std::shared_ptr<Identity> newid(new Identity(Identity::newref(), id.get()->fieldloc(), id.get()->chunkdepth(), id.get()->indexdepth() + 1, content_.get()->length()));
    Error err = awkward_identity_from_listfoffsets(length(), id.get()->keydepth(), offsets_.ptr().get(), id.get()->ptr().get(), content_.get()->length(), newid.get()->ptr().get());
    HANDLE_ERROR(err);
    content_.get()->setid(newid);
  }
  id_ = id;
}

const std::string ListOffsetArray::repr(const std::string indent, const std::string pre, const std::string post) const {
  std::stringstream out;
  out << indent << pre << "<ListOffsetArray>\n";
  if (id_.get() != nullptr) {
    out << id_.get()->repr(indent + std::string("    "), "", "\n");
  }
  out << offsets_.repr(indent + std::string("    "), "<offsets>", "</offsets>\n");
  out << content_.get()->repr(indent + std::string("    "), "<content>", "</content>\n");
  out << indent << "</ListOffsetArray>" << post;
  return out.str();
}

IndexType ListOffsetArray::length() const {
  return offsets_.length() - 1;
}

std::shared_ptr<Content> ListOffsetArray::shallow_copy() const {
  return std::shared_ptr<Content>(new ListOffsetArray(id_, offsets_, content_));
}

std::shared_ptr<Content> ListOffsetArray::get(IndexType at) const {
  IndexType start = offsets_.get(at);
  IndexType stop = offsets_.get(at + 1);
  return content_.get()->slice(start, stop);
}

std::shared_ptr<Content> ListOffsetArray::slice(IndexType start, IndexType stop) const {
  std::shared_ptr<Identity> id;
  if (id_.get() != nullptr) {
    id = id_.get()->slice(start, stop);
  }
  return std::shared_ptr<Content>(new ListOffsetArray(id, offsets_.slice(start, stop + 1), content_));
}
