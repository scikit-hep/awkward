// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

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
        !form_.get()->equal(out.get()->form(), true, true)) {
      throw std::invalid_argument(
          std::string("generated array does not conform to expected form:\n\n")
          + form_.get()->tostring() + std::string("\n\nbut generated:\n\n")
          + out.get()->form().get()->tostring());
    }
    return out;
  }
}
