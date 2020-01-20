// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_UNKNOWNTYPE_H_
#define AWKWARD_UNKNOWNTYPE_H_

#include "awkward/type/Type.h"

namespace awkward {
  class UnknownType: public Type {
  public:
    UnknownType(const util::Parameters& parameters);

    std::string tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const override;
    const std::shared_ptr<Type> shallow_copy() const override;
    bool equal(const std::shared_ptr<Type>& other, bool check_parameters) const override;
    int64_t numfields() const override;
    int64_t fieldindex(const std::string& key) const override;
    const std::string key(int64_t fieldindex) const override;
    bool haskey(const std::string& key) const override;
    const std::vector<std::string> keys() const override;
    const std::shared_ptr<Content> empty() const override;

  private:
  };
}

#endif // AWKWARD_LISTTYPE_H_
