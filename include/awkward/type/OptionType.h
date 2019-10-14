// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_OPTIONTYPE_H_
#define AWKWARD_OPTIONTYPE_H_

#include "awkward/type/Type.h"

namespace awkward {
  class OptionType: public Type {
  public:
    OptionType(const std::shared_ptr<Type> type): type_(type) { }

    virtual std::string tostring_part(std::string indent, std::string pre, std::string post) const;
    virtual bool equal(std::shared_ptr<Type> other) const;

  std::shared_ptr<Type> type() const;

  private:
    std::shared_ptr<Type> type_;
  };
}

#endif // AWKWARD_ARRAYTYPE_H_
