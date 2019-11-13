// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_REGULARTYPE_H_
#define AWKWARD_REGULARTYPE_H_

#include <vector>

#include "awkward/type/Type.h"

namespace awkward {
  class RegularType: public Type {
  public:
    RegularType(const std::vector<int64_t> shape, const std::shared_ptr<Type> type): shape_(shape), type_(type) { }

    virtual std::string tostring_part(std::string indent, std::string pre, std::string post) const;
    virtual const std::shared_ptr<Type> shallow_copy() const;
    virtual bool compatible(std::shared_ptr<Type> other) const;

    const std::vector<int64_t> shape() const;
    const std::shared_ptr<Type> type() const;

  private:
    const std::vector<int64_t> shape_;
    const std::shared_ptr<Type> type_;
  };
}

#endif // AWKWARD_REGULARTYPE_H_
