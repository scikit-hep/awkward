// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_REGULARTYPE_H_
#define AWKWARD_REGULARTYPE_H_

#include <vector>

#include "awkward/type/Type.h"

namespace awkward {
  class RegularType: public Type {
  public:
    RegularType(const Parameters& parameters, const std::shared_ptr<Type>& type, int64_t size)
        : Type(parameters)
        , type_(type)
        , size_(size) { }

    std::string tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const override;
    const std::shared_ptr<Type> shallow_copy() const override;
    bool equal(const std::shared_ptr<Type>& other, bool check_parameters) const override;
    int64_t numfields() const override;
    int64_t fieldindex(const std::string& key) const override;
    const std::string key(int64_t fieldindex) const override;
    bool haskey(const std::string& key) const override;
    const std::vector<std::string> keyaliases(int64_t fieldindex) const override;
    const std::vector<std::string> keyaliases(const std::string& key) const override;
    const std::vector<std::string> keys() const override;

    const std::shared_ptr<Type> type() const;
    int64_t size() const;

  private:
    const std::shared_ptr<Type> type_;
    const int64_t size_;
  };
}

#endif // AWKWARD_REGULARTYPE_H_
