// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include "sstream"

#include "awkward/array/VirtualArray.h"

#include "awkward/virtual/ArrayGenerator.h"

namespace awkward {
  ArrayGenerator::ArrayGenerator(const FormPtr& form, int64_t length)
      : form_(form)
      , length_(length) { }

  const FormPtr
  ArrayGenerator::form() const {
    return form_;
  }

  int64_t
  ArrayGenerator::length() const {
    return length_;
  }

  const ContentPtr
  ArrayGenerator::generate_and_check() const {
    ContentPtr out = generate();
    if (length_ >= 0  &&  length_ != out.get()->length()) {
      throw std::invalid_argument(
          std::string("generated array does not have the expected length: ")
          + std::to_string(length_) + std::string(" but generated ")
          + std::to_string(out.get()->length()));
    }
    if (form_.get() != nullptr  &&
        !form_.get()->equal(out.get()->form(true), true, true)) {
      throw std::invalid_argument(
          std::string("generated array does not conform to expected form:\n\n")
          + form_.get()->tostring() + std::string("\n\nbut generated:\n\n")
          + out.get()->form(true).get()->tostring());
    }
    return out;
  }

  SliceGenerator::SliceGenerator(const FormPtr& form,
                                 int64_t length,
                                 const ContentPtr& content,
                                 const Slice& slice)
      : ArrayGenerator(form, length)
      , content_(content)
      , slice_(slice) { }

  const ContentPtr
  SliceGenerator::content() const {
    return content_;
  }

  const Slice
  SliceGenerator::slice() const {
    return slice_;
  }

  const ContentPtr
  SliceGenerator::generate() const {
    if (slice_.length() == 1) {
      SliceItemPtr head = slice_.head();
      if (SliceRange* raw = dynamic_cast<SliceRange*>(head.get())) {
        if (raw->step() == 1) {
          if (VirtualArray* a = dynamic_cast<VirtualArray*>(content_.get())) {
            return a->array().get()->getitem_range(raw->start(), raw->stop());
          }
          else {
            return content_.get()->getitem_range(raw->start(), raw->stop());
          }
        }
      }
    }
    if (VirtualArray* a = dynamic_cast<VirtualArray*>(content_.get())) {
      return a->array().get()->getitem(slice_);
    }
    else {
      return content_.get()->getitem(slice_);
    }
  }

  const std::string
  SliceGenerator::tostring_part(const std::string& indent,
                                const std::string& pre,
                                const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<SliceGenerator>\n";
    out << indent << "    <slice>" << slice_.tostring() << "</slice>\n";
    out << content_.get()->tostring_part(
             indent + std::string("    "), "<content>", "</content>\n");
    out << indent << "</SliceGenerator>" << post;
    return out.str();
  }
}
