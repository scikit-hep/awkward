// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "sstream"

#include "awkward/virtual/ArrayGenerator.h"

namespace awkward {
  ArrayGenerator::ArrayGenerator(const FormPtr& form, int64_t length)
      : form_(form)
      , length_(length) { }

  ArrayGenerator::~ArrayGenerator() = default;

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
                                 const ArrayGeneratorPtr& generator,
                                 const Slice& slice)
      : ArrayGenerator(form, length)
      , generator_(generator)
      , slice_(slice) { }

  const ArrayGeneratorPtr
  SliceGenerator::generator() const {
    return generator_;
  }

  const Slice
  SliceGenerator::slice() const {
    return slice_;
  }

  const ContentPtr
  SliceGenerator::generate() const {
    ContentPtr inner = generator_.get()->generate();
    if (slice_.length() == 1) {
      SliceItemPtr head = slice_.head();
      if (SliceRange* raw = dynamic_cast<SliceRange*>(head.get())) {
        if (raw->step() == 1) {
          return inner.get()->getitem_range(raw->start(), raw->stop());
        }
      }
    }
    return inner.get()->getitem(slice_);
  }

  const std::string
  SliceGenerator::tostring_part(const std::string& indent,
                                const std::string& pre,
                                const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<SliceGenerator>\n";
    out << indent << "    <slice>" << slice_.tostring() << "</slice>\n";
    out << generator_.get()->tostring_part(
             indent + std::string("    "), "<generator>", "</generator>\n");
    out << indent << "</SliceGenerator>" << post;
    return out.str();
  }
}
