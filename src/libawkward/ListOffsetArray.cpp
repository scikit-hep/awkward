// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/ListOffsetArray.h"

using namespace awkward;

void ListOffsetArray::setid() {
  throw std::runtime_error("not implemented");
}

const std::string ListOffsetArray::repr(const std::string indent, const std::string pre, const std::string post) const {
  std::stringstream out;
  out << indent << pre << "<ListOffsetArray>" << std::endl;
  out << offsets_.repr(indent + std::string("    "), "<offsets>", "</offsets>\n");
  out << content_.get()->repr(indent + std::string("    "), "<content>", "</content>\n");
  out << indent << "</ListOffsetArray>" << post;
  return out.str();
}

IndexType ListOffsetArray::length() const {
  return offsets_.length() - 1;
}

std::shared_ptr<Content> ListOffsetArray::shallow_copy() const {
  return std::shared_ptr<Content>(new ListOffsetArray(offsets_, content_));
}

std::shared_ptr<Content> ListOffsetArray::get(IndexType at) const {
  IndexType start = offsets_.get(at);
  IndexType stop = offsets_.get(at + 1);
  return content_.get()->slice(start, stop);
}

std::shared_ptr<Content> ListOffsetArray::slice(IndexType start, IndexType stop) const {
  return std::shared_ptr<Content>(new ListOffsetArray(offsets_.slice(start, stop + 1), content_));
}
