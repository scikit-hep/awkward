// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_ARRAYTYPE_H_
#define AWKWARD_ARRAYTYPE_H_

#include "awkward/type/Type.h"

namespace awkward {
  class ArrayType: public Type {
  public:
    ArrayType(int64_t length, const std::shared_ptr<Type> type): length_(length), type_(type) { }

    int64_t length() const { return length_; }
    std::shared_ptr<Type> type() const { return type_; }

    virtual std::string tostring_part(std::string indent, std::string pre, std::string post) const;
    virtual bool equal(std::shared_ptr<Type> other) const;

  private:
    int64_t length_;
    std::shared_ptr<Type> type_;
  };
}

#endif // AWKWARD_ARRAYTYPE_H_
