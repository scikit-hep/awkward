// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_REDUCER_H_
#define AWKWARD_REDUCER_H_

#include <memory>

#include "awkward/Index.h"

namespace awkward {
  class Reducer {
  public:
    virtual const std::shared_ptr<bool> apply_bool(const std::shared_ptr<bool>& data, const Index64& parents, int64_t outlength) = 0;
    virtual const std::shared_ptr<int8_t> apply_int8(const std::shared_ptr<int8_t>& data, const Index64& parents, int64_t outlength) = 0;
    virtual const std::shared_ptr<uint8_t> apply_uint8(const std::shared_ptr<uint8_t>& data, const Index64& parents, int64_t outlength) = 0;
    virtual const std::shared_ptr<int16_t> apply_int16(const std::shared_ptr<int16_t>& data, const Index64& parents, int64_t outlength) = 0;
    virtual const std::shared_ptr<uint16_t> apply_uint16(const std::shared_ptr<uint16_t>& data, const Index64& parents, int64_t outlength) = 0;
    virtual const std::shared_ptr<int32_t> apply_int32(const std::shared_ptr<int32_t>& data, const Index64& parents, int64_t outlength) = 0;
    virtual const std::shared_ptr<uint32_t> apply_uint32(const std::shared_ptr<uint32_t>& data, const Index64& parents, int64_t outlength) = 0;
    virtual const std::shared_ptr<int64_t> apply_int64(const std::shared_ptr<int64_t>& data, const Index64& parents, int64_t outlength) = 0;
    virtual const std::shared_ptr<uint64_t> apply_uint64(const std::shared_ptr<uint64_t>& data, const Index64& parents, int64_t outlength) = 0;
    virtual const std::shared_ptr<float> apply_float32(const std::shared_ptr<float>& data, const Index64& parents, int64_t outlength) = 0;
    virtual const std::shared_ptr<double> apply_float64(const std::shared_ptr<double>& data, const Index64& parents, int64_t outlength) = 0;
  };
}

#endif // AWKWARD_REDUCER_H_
