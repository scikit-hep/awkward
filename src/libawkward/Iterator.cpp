// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/Iterator.h"

using namespace awkward;

const std::string Iterator::repr(const std::string indent, const std::string pre, const std::string post) const {
  std::stringstream out;
  out << indent << pre << "<Iterator where=\"" << where_ << "\">\n";
  out << content_.get()->repr(indent + std::string("    "), "", "\n");
  out << indent << "</Iterator>" << post;
  return out.str();
}
