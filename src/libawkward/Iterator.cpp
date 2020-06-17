// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include <sstream>

#include "awkward/array/VirtualArray.h"

#include "awkward/Iterator.h"

namespace awkward {
  const ContentPtr nonvirtual(const ContentPtr& content) {
    if (VirtualArray* raw = dynamic_cast<VirtualArray*>(content.get())) {
      return raw->array();
    }
    else {
      return content;
    }
  }

  Iterator::Iterator(const ContentPtr& content)
    : content_(nonvirtual(content))
    , at_(0) {
    content.get()->check_for_iteration();
  }

  const ContentPtr
  Iterator::content() const {
    return content_;
  }

  const int64_t
  Iterator::at() const {
    return at_;
  }

  const bool
  Iterator::isdone() const {
    return at_ >= content_.get()->length();
  }

  const ContentPtr
  Iterator::next() {
    return content_.get()->getitem_at_nowrap(at_++);
  }

  const std::string
  Iterator::tostring_part(const std::string& indent,
                          const std::string& pre,
                          const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<Iterator at=\"" << at_ << "\">\n";
    out << content_.get()->tostring_part(
             indent + std::string("    "), "", "\n");
    out << indent << "</Iterator>" << post;
    return out.str();
  }

  const std::string
  Iterator::tostring() const {
    return tostring_part("", "", "");
  }
}
