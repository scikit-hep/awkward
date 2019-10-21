// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_ARRAYTYPE_H_
#define AWKWARD_ARRAYTYPE_H_

#include "awkward/type/Type.h"

namespace awkward {
  class ArrayType: public Type {
  public:
    ArrayType(int64_t length, const std::shared_ptr<Type> type): length_(length), type_(type) { }

    virtual std::string tostring_part(std::string indent, std::string pre, std::string post) const;
    virtual const std::shared_ptr<Type> shallow_copy() const;
    virtual bool equal(std::shared_ptr<Type> other) const;

    int64_t length() const;
    const std::shared_ptr<Type> type() const;

  private:
    std::shared_ptr<Type> type_;
    int64_t length_;
  };
}

#endif // AWKWARD_ARRAYTYPE_H_
