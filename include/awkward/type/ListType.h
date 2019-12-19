// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_LISTTYPE_H_
#define AWKWARD_LISTTYPE_H_

#include "awkward/type/Type.h"

namespace awkward {
  class ListType: public Type {
  public:
    ListType(const Parameters& parameters, const std::shared_ptr<Type>& type);

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

  private:
    const std::shared_ptr<Type> type_;
  };
}

#endif // AWKWARD_LISTTYPE_H_
