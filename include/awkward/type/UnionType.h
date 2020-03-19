// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_UNIONTYPE_H_
#define AWKWARD_UNIONTYPE_H_

#include <vector>

#include "awkward/type/Type.h"

namespace awkward {
  class EXPORT_SYMBOL UnionType: public Type {
  public:
    UnionType(const util::Parameters& parameters, const std::string& typestr, const std::vector<TypePtr>& types);

    std::string tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const override;
    const TypePtr shallow_copy() const override;
    bool equal(const TypePtr& other, bool check_parameters) const override;
    int64_t numfields() const override;
    int64_t fieldindex(const std::string& key) const override;
    const std::string key(int64_t fieldindex) const override;
    bool haskey(const std::string& key) const override;
    const std::vector<std::string> keys() const override;
    const ContentPtr empty() const override;

    int64_t numtypes() const;
    const std::vector<TypePtr> types() const;
    const TypePtr type(int64_t index) const;

  private:
    const std::vector<TypePtr> types_;
  };
}

#endif // AWKWARD_OPTIONTYPE_H_
