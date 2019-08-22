// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/ListOffsetArray.h"

using namespace awkward;

const std::string ListOffsetArray::repr(const std::string indent, const std::string pre, const std::string post) const {
  std::stringstream out;
  out << indent << pre << "<ListOffsetArray>" << std::endl;
  out << offsets_.repr(indent + std::string("    "), "<offsets>", "</offsets>\n");
  out << content_.get()->repr(indent + std::string("    "), "<content>", "</content>\n");
  out << indent << "</ListOffsetArray>" << post;
  return out.str();
}

std::shared_ptr<Content> ListOffsetArray::get(AtType at) const {
  assert(false);
}

std::shared_ptr<Content> ListOffsetArray::slice(AtType start, AtType stop) const {
  assert(false);
}
