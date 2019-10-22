// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_PRIMITIVETYPE_H_
#define AWKWARD_PRIMITIVETYPE_H_

#include "awkward/type/Type.h"

namespace awkward {
  class PrimitiveType: public Type {
  public:
    enum DType {
      boolean,
      int8,
      int16,
      int32,
      int64,
      uint8,
      uint16,
      uint32,
      uint64,
      float32,
      float64,
      numtypes
    };

    PrimitiveType(DType dtype): dtype_(dtype) { }

    virtual std::string tostring_part(std::string indent, std::string pre, std::string post) const;
    virtual const std::shared_ptr<Type> shallow_copy() const;
    virtual bool equal(std::shared_ptr<Type> other) const;

  private:
    DType dtype_;
  };
}

#endif // AWKWARD_PRIMITIVETYPE_H_
