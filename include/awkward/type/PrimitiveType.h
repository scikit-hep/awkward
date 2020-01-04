// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_PRIMITIVETYPE_H_
#define AWKWARD_PRIMITIVETYPE_H_

#include "awkward/type/Type.h"

namespace awkward {
  class PrimitiveType: public Type {
  public:
    enum DType {
      boolean,
      int8,
      int16,
      int32,
      int64,
      uint8,
      uint16,
      uint32,
      uint64,
      float32,
      float64,
      numtypes
    };

    PrimitiveType(const util::Parameters& parameters, DType dtype);

    std::string tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const override;
    const std::shared_ptr<Type> shallow_copy() const override;
    bool equal(const std::shared_ptr<Type>& other, bool check_parameters) const override;
    int64_t numfields() const override;
    int64_t fieldindex(const std::string& key) const override;
    const std::string key(int64_t fieldindex) const override;
    bool haskey(const std::string& key) const override;
    const std::vector<std::string> keys() const override;
    const std::shared_ptr<Content> empty() const override;

  const DType dtype() const;

  private:
    const DType dtype_;
  };
}

#endif // AWKWARD_PRIMITIVETYPE_H_
