// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <sstream>

#include "awkward/array/NumpyArray.h"
#include "awkward/type/UnknownType.h"
#include "awkward/type/OptionType.h"

#include "awkward/type/PrimitiveType.h"

namespace awkward {
  PrimitiveType::PrimitiveType(const util::Parameters& parameters, DType dtype)
      : Type(parameters)
      , dtype_(dtype) { }

  std::string PrimitiveType::tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const {
    std::string typestr;
    if (get_typestr(typestr)) {
      return typestr;
    }

    std::stringstream out;
    std::string s;
    switch (dtype_) {
      case boolean: s = "bool"; break;
      case int8:    s = "int8"; break;
      case int16:   s = "int16"; break;
      case int32:   s = "int32"; break;
      case int64:   s = "int64"; break;
      case uint8:   s = "uint8"; break;
      case uint16:  s = "uint16"; break;
      case uint32:  s = "uint32"; break;
      case uint64:  s = "uint64"; break;
      case float32: s = "float32"; break;
      case float64: s = "float64"; break;
      default:      s = "unknown"; break;
    }
    if (parameters_.empty()) {
      out << indent << pre << s << post;
    }
    else {
      out << indent << pre << s << "[" << string_parameters() << "]" << post;
    }
    return out.str();
  }

  const std::shared_ptr<Type> PrimitiveType::shallow_copy() const {
    return std::make_shared<PrimitiveType>(parameters_, dtype_);
  }

  bool PrimitiveType::equal(const std::shared_ptr<Type>& other, bool check_parameters) const {
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

  int64_t PrimitiveType::numfields() const {
    return -1;
  }

  int64_t PrimitiveType::fieldindex(const std::string& key) const {
    throw std::invalid_argument("type contains no Records");
  }

  const std::string PrimitiveType::key(int64_t fieldindex) const {
    throw std::invalid_argument("type contains no Records");
  }

  bool PrimitiveType::haskey(const std::string& key) const {
    throw std::invalid_argument("type contains no Records");
  }

  const std::vector<std::string> PrimitiveType::keys() const {
    throw std::invalid_argument("type contains no Records");
  }

  const std::shared_ptr<Content> PrimitiveType::empty() const {
    std::shared_ptr<void> ptr(new uint8_t[0], util::array_deleter<uint8_t>());
    std::vector<ssize_t> shape({ 0 });
    std::vector<ssize_t> strides({ 0 });
    ssize_t itemsize;
    std::string format;
    switch (dtype_) {
      case boolean: itemsize = 1; format = "?"; break;
      case int8:    itemsize = 1; format = "b"; break;
      case uint8:   itemsize = 1; format = "B"; break;
      case int16:   itemsize = 2; format = "h"; break;
      case uint16:  itemsize = 2; format = "H"; break;
#if defined _MSC_VER || defined __i386__
      case int32:   itemsize = 4; format = "l"; break;
      case uint32:  itemsize = 4; format = "L"; break;
      case int64:   itemsize = 8; format = "q"; break;
      case uint64:  itemsize = 8; format = "Q"; break;
#else
      case int32:   itemsize = 4; format = "i"; break;
      case uint32:  itemsize = 4; format = "I"; break;
      case int64:   itemsize = 8; format = "l"; break;
      case uint64:  itemsize = 8; format = "L"; break;
#endif
      case float32: itemsize = 4; format = "f"; break;
      case float64: itemsize = 8; format = "d"; break;
      default: throw std::runtime_error(std::string("unexpected dtype: ") + std::to_string(dtype_));
    }
    return std::make_shared<NumpyArray>(Identities::none(), parameters_, ptr, shape, strides, 0, itemsize, format);
  }

  const PrimitiveType::DType PrimitiveType::dtype() const {
    return dtype_;
  }
}
