// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/type/UnknownType.cpp", line)

#include <string>
#include <sstream>

#include "awkward/array/EmptyArray.h"
#include "awkward/type/UnknownType.h"

namespace awkward {
  UnknownType::UnknownType(const util::Parameters& parameters,
                           const std::string& typestr)
      : Type(parameters, typestr) { }

  std::string
  UnknownType::tostring_part(const std::string& indent,
                             const std::string& pre,
                             const std::string& post) const {
    std::string typestr;
    if (get_typestr(typestr)) {
      return wrap_categorical(typestr);
    }

    std::stringstream out;
    if (parameters_empty()) {
      out << indent << pre << "unknown" << post;
    }
    else {
      out << indent << pre << "unknown[" << string_parameters() << "]" << post;
    }
    return wrap_categorical(out.str());
  }

  const TypePtr
  UnknownType::shallow_copy() const {
    return std::make_shared<UnknownType>(parameters_, typestr_);
  }

  bool
  UnknownType::equal(const TypePtr& other, bool check_parameters) const {
    if (UnknownType* t = dynamic_cast<UnknownType*>(other.get())) {
      if (check_parameters  &&  !parameters_equal(other.get()->parameters())) {
        return false;
      }
      return true;
    }
    else {
      return false;
    }
  }

  int64_t
  UnknownType::numfields() const {
    return -1;
  }

  int64_t
  UnknownType::fieldindex(const std::string& key) const {
    throw std::invalid_argument(
      std::string("type contains no Records") + FILENAME(__LINE__));
  }

  const std::string
  UnknownType::key(int64_t fieldindex) const {
    throw std::invalid_argument(
      std::string("type contains no Records") + FILENAME(__LINE__));
  }

  bool
  UnknownType::haskey(const std::string& key) const {
    throw std::invalid_argument(
      std::string("type contains no Records") + FILENAME(__LINE__));
  }

  const std::vector<std::string>
  UnknownType::keys() const {
    throw std::invalid_argument(
      std::string("type contains no Records") + FILENAME(__LINE__));
  }

  const ContentPtr
  UnknownType::empty() const {
    return std::make_shared<EmptyArray>(Identities::none(), parameters_);
  }
}
