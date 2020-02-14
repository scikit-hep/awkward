// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/reducers.h"

#include "awkward/Reducer.h"

namespace awkward {
  const std::shared_ptr<bool> ReducerProd::apply_bool(const std::shared_ptr<bool>& data, int64_t offset, const Index64& parents, int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[outlength], util::array_deleter<bool>());
    struct Error err = awkward_reduce_prod_bool_64(
      ptr.get(),
      data.get(),
      offset,
      parents.ptr().get(),
      parents.offset(),
      outlength);
    util::handle_error(err, "'prod'", nullptr);
    return ptr;
  }

  const std::shared_ptr<int8_t> ReducerProd::apply_int8(const std::shared_ptr<int8_t>& data, int64_t offset, const Index64& parents, int64_t outlength) const {
    throw std::runtime_error("FIXME: ReducerProd::apply_int8");
  }

  const std::shared_ptr<uint8_t> ReducerProd::apply_uint8(const std::shared_ptr<uint8_t>& data, int64_t offset, const Index64& parents, int64_t outlength) const {
    throw std::runtime_error("FIXME: ReducerProd::apply_uint8");
  }

  const std::shared_ptr<int16_t> ReducerProd::apply_int16(const std::shared_ptr<int16_t>& data, int64_t offset, const Index64& parents, int64_t outlength) const {
    throw std::runtime_error("FIXME: ReducerProd::apply_int16");
  }

  const std::shared_ptr<uint16_t> ReducerProd::apply_uint16(const std::shared_ptr<uint16_t>& data, int64_t offset, const Index64& parents, int64_t outlength) const {
    throw std::runtime_error("FIXME: ReducerProd::apply_uint16");
  }

  const std::shared_ptr<int32_t> ReducerProd::apply_int32(const std::shared_ptr<int32_t>& data, int64_t offset, const Index64& parents, int64_t outlength) const {
    throw std::runtime_error("FIXME: ReducerProd::apply_int32");
  }

  const std::shared_ptr<uint32_t> ReducerProd::apply_uint32(const std::shared_ptr<uint32_t>& data, int64_t offset, const Index64& parents, int64_t outlength) const {
    throw std::runtime_error("FIXME: ReducerProd::apply_uint32");
  }

  const std::shared_ptr<int64_t> ReducerProd::apply_int64(const std::shared_ptr<int64_t>& data, int64_t offset, const Index64& parents, int64_t outlength) const {
    throw std::runtime_error("FIXME: ReducerProd::apply_int64");
  }

  const std::shared_ptr<uint64_t> ReducerProd::apply_uint64(const std::shared_ptr<uint64_t>& data, int64_t offset, const Index64& parents, int64_t outlength) const {
    throw std::runtime_error("FIXME: ReducerProd::apply_uint64");
  }

  const std::shared_ptr<float> ReducerProd::apply_float32(const std::shared_ptr<float>& data, int64_t offset, const Index64& parents, int64_t outlength) const {
    throw std::runtime_error("FIXME: ReducerProd::apply_float32");
  }

  const std::shared_ptr<double> ReducerProd::apply_float64(const std::shared_ptr<double>& data, int64_t offset, const Index64& parents, int64_t outlength) const {
    throw std::runtime_error("FIXME: ReducerProd::apply_float64");
  }
}
