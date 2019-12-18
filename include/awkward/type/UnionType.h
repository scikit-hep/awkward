// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_UNIONTYPE_H_
#define AWKWARD_UNIONTYPE_H_

#include <vector>

#include "awkward/type/Type.h"

namespace awkward {
  class UnionType: public Type {
  public:
    UnionType(const Parameters& parameters, const std::vector<std::shared_ptr<Type>>& types): Type(parameters), types_(types) { }

    std::string tostring_part(std::string indent, std::string pre, std::string post) const override;
    const std::shared_ptr<Type> shallow_copy() const override;
    bool equal(const std::shared_ptr<Type> other, bool check_parameters) const override;
    std::shared_ptr<Type> level() const override;
    std::shared_ptr<Type> inner() const override;
    std::shared_ptr<Type> inner(const std::string& key) const override;
    int64_t numfields() const override;
    int64_t fieldindex(const std::string& key) const override;
    const std::string key(int64_t fieldindex) const override;
    bool haskey(const std::string& key) const override;
    const std::vector<std::string> keyaliases(int64_t fieldindex) const override;
    const std::vector<std::string> keyaliases(const std::string& key) const override;
    const std::vector<std::string> keys() const override;

    int64_t numtypes() const;
    const std::vector<std::shared_ptr<Type>> types() const;
    const std::shared_ptr<Type> type(int64_t index) const;

  private:
    const std::vector<std::shared_ptr<Type>> types_;
  };
}

#endif // AWKWARD_OPTIONTYPE_H_
