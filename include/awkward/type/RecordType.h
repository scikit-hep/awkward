// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_RECORDTYPE_H_
#define AWKWARD_RECORDTYPE_H_

#include <vector>
#include <string>
#include <unordered_map>

#include "awkward/util.h"

#include "awkward/type/Type.h"

namespace awkward {
  class EXPORT_SYMBOL RecordType: public Type {
  public:
    RecordType(const util::Parameters& parameters,
               const std::string& typestr,
               const std::vector<TypePtr>& types,
               const util::RecordLookupPtr& recordlookup);

    RecordType(const util::Parameters& parameters,
               const std::string& typestr,
               const std::vector<TypePtr>& types);

    const std::vector<TypePtr>
      types() const;

    const util::RecordLookupPtr
      recordlookup() const;

    bool
      istuple() const;

    std::string
      tostring_part(const std::string& indent,
                    const std::string& pre,
                    const std::string& post) const override;

    const TypePtr
      shallow_copy() const override;

    bool
      equal(const TypePtr& other, bool check_parameters) const override;

    int64_t
      numfields() const override;

    int64_t
      fieldindex(const std::string& key) const override;

    const std::string
      key(int64_t fieldindex) const override;

    bool
      haskey(const std::string& key) const override;

    const std::vector<std::string>
      keys() const override;

    const ContentPtr
      empty() const override;

    const TypePtr
      field(int64_t fieldindex) const;

    const TypePtr
      field(const std::string& key) const;

    const std::vector<TypePtr>
      fields() const;

    const std::vector<std::pair<std::string, TypePtr>>
      fielditems() const;

    const TypePtr
      astuple() const;

  private:
    const std::vector<TypePtr> types_;
    const util::RecordLookupPtr recordlookup_;
  };
}

#endif // AWKWARD_RECORDTYPE_H_
