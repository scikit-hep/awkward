// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_UNIONTYPE_H_
#define AWKWARD_UNIONTYPE_H_

#include <vector>

#include "awkward/type/Type.h"

namespace awkward {
  class UnionType: public Type {
  public:
    UnionType(const std::vector<std::shared_ptr<Type>> types): types_(types) { }

    virtual std::string tostring_part(std::string indent, std::string pre, std::string post) const;
    virtual bool equal(std::shared_ptr<Type> other) const;

    int64_t numtypes() const;
    const std::vector<std::shared_ptr<Type>> types() const;
    const std::shared_ptr<Type> type(int64_t i) const;

  private:
    std::vector<std::shared_ptr<Type>> types_;
  };
}

#endif // AWKWARD_OPTIONTYPE_H_
