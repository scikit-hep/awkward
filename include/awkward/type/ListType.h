// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_LISTTYPE_H_
#define AWKWARD_LISTTYPE_H_

#include "awkward/type/Type.h"

namespace awkward {
  class ListType: public Type {
  public:
    ListType(const std::shared_ptr<Type> type): type_(type) { }

    virtual std::string tostring_part(std::string indent, std::string pre, std::string post) const;
    virtual const std::shared_ptr<Type> shallow_copy() const;
    virtual bool equal(std::shared_ptr<Type> other) const;
    virtual bool compatible(std::shared_ptr<Type> other, bool bool_is_int, bool int_is_float, bool ignore_null, bool unknown_is_anything) const;

  const std::shared_ptr<Type> type() const;

  private:
    const std::shared_ptr<Type> type_;
  };
}

#endif // AWKWARD_LISTTYPE_H_
