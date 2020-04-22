// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/virtual/ArrayGenerator.h"

namespace awkward {
  ArrayGenerator::ArrayGenerator(const FormPtr& form)
      : form_(form) { }

  const FormPtr
  ArrayGenerator::form() const {
    return form_;
  }

  const ContentPtr
  ArrayGenerator::generate_and_check() const {
    throw std::invalid_argument(
        std::string("generated array does not conform to expected form:\n\n")
        + form_.get()->tostring() + std::string("\n\nbut generated:\n\n")
        );  // HERE
  }

}
