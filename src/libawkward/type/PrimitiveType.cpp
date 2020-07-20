// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include <sstream>

#include "awkward/array/NumpyArray.h"
#include "awkward/type/UnknownType.h"
#include "awkward/type/OptionType.h"

#include "awkward/type/PrimitiveType.h"

namespace awkward {
  PrimitiveType::PrimitiveType(const util::Parameters& parameters,
                               const std::string& typestr,
                               util::dtype dtype)
      : Type(parameters, typestr)
      , dtype_(dtype) { }

  std::string
  PrimitiveType::tostring_part(const std::string& indent,
                               const std::string& pre,
                               const std::string& post) const {
    std::string typestr;
    if (get_typestr(typestr)) {
      return typestr;
    }

    std::stringstream out;
    std::string s = util::dtype_to_name(dtype_);
    if (parameters_.empty()) {
      out << indent << pre << s << post;
    }
    else {
      out << indent << pre << s << "[" << string_parameters() << "]" << post;
    }
    return out.str();
  }

  const TypePtr
  PrimitiveType::shallow_copy() const {
    return std::make_shared<PrimitiveType>(parameters_, typestr_, dtype_);
  }

  bool
  PrimitiveType::equal(const TypePtr& other, bool check_parameters) const {
    if (PrimitiveType* t = dynamic_cast<PrimitiveType*>(other.get())) {
      if (check_parameters  &&  !parameters_equal(other.get()->parameters())) {
        return false;
      }
      return dtype_ == t->dtype_;
    }
    else {
      return false;
    }
  }

  int64_t
  PrimitiveType::numfields() const {
    return -1;
  }

  int64_t
  PrimitiveType::fieldindex(const std::string& key) const {
    throw std::invalid_argument("type contains no Records");
  }

  const std::string
  PrimitiveType::key(int64_t fieldindex) const {
    throw std::invalid_argument("type contains no Records");
  }

  bool
  PrimitiveType::haskey(const std::string& key) const {
    throw std::invalid_argument("type contains no Records");
  }

  const std::vector<std::string>
  PrimitiveType::keys() const {
    throw std::invalid_argument("type contains no Records");
  }

  const ContentPtr
  PrimitiveType::empty() const {
    std::shared_ptr<void> ptr(new uint8_t[0], kernel::array_deleter<uint8_t>());
    std::vector<ssize_t> shape({ 0 });
    std::vector<ssize_t> strides({ 0 });
    std::string format = util::dtype_to_format(dtype_);
    if (format.length() == 0) {
      throw std::invalid_argument(
          "cannot create an empty array of unknown PrimitiveType");
    }
    return std::make_shared<NumpyArray>(Identities::none(),
                                        parameters_,
                                        ptr,
                                        shape,
                                        strides,
                                        0,
                                        util::dtype_to_itemsize(dtype_),
                                        format,
                                        dtype_);
  }

  util::dtype
  PrimitiveType::dtype() const {
    return dtype_;
  }
}
