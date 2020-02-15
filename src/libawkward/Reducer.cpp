// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include "awkward/cpu-kernels/reducers.h"

#include "awkward/Reducer.h"

namespace awkward {
  const std::string ReducerProd::preferred_type() const {
#if defined _MSC_VER || defined __i386__
    return "q";
#else
    return "l";
#endif
  }

  ssize_t ReducerProd::preferred_typesize() const {
    return 8;
  }

  const std::shared_ptr<bool> ReducerProd::apply_bool(const bool* data, int64_t offset, const Index64& parents, int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength], util::array_deleter<bool>());
    struct Error err = awkward_reduce_prod_bool_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, "'prod'", nullptr);
    return ptr;
  }

  const std::shared_ptr<int8_t> ReducerProd::apply_int8(const int8_t* data, int64_t offset, const Index64& parents, int64_t outlength) const {
    std::shared_ptr<int8_t> ptr(new int8_t[(size_t)outlength], util::array_deleter<int8_t>());
    struct Error err = awkward_reduce_prod_int8_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, "'prod'", nullptr);
    return ptr;
  }

  const std::shared_ptr<uint8_t> ReducerProd::apply_uint8(const uint8_t* data, int64_t offset, const Index64& parents, int64_t outlength) const {
    std::shared_ptr<uint8_t> ptr(new uint8_t[(size_t)outlength], util::array_deleter<uint8_t>());
    struct Error err = awkward_reduce_prod_uint8_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, "'prod'", nullptr);
    return ptr;
  }

  const std::shared_ptr<int16_t> ReducerProd::apply_int16(const int16_t* data, int64_t offset, const Index64& parents, int64_t outlength) const {
    std::shared_ptr<int16_t> ptr(new int16_t[(size_t)outlength], util::array_deleter<int16_t>());
    struct Error err = awkward_reduce_prod_int16_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, "'prod'", nullptr);
    return ptr;
  }

  const std::shared_ptr<uint16_t> ReducerProd::apply_uint16(const uint16_t* data, int64_t offset, const Index64& parents, int64_t outlength) const {
    std::shared_ptr<uint16_t> ptr(new uint16_t[(size_t)outlength], util::array_deleter<uint16_t>());
    struct Error err = awkward_reduce_prod_uint16_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, "'prod'", nullptr);
    return ptr;
  }

  const std::shared_ptr<int32_t> ReducerProd::apply_int32(const int32_t* data, int64_t offset, const Index64& parents, int64_t outlength) const {
    std::shared_ptr<int32_t> ptr(new int32_t[(size_t)outlength], util::array_deleter<int32_t>());
    struct Error err = awkward_reduce_prod_int32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, "'prod'", nullptr);
    return ptr;
  }

  const std::shared_ptr<uint32_t> ReducerProd::apply_uint32(const uint32_t* data, int64_t offset, const Index64& parents, int64_t outlength) const {
    std::shared_ptr<uint32_t> ptr(new uint32_t[(size_t)outlength], util::array_deleter<uint32_t>());
    struct Error err = awkward_reduce_prod_uint32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, "'prod'", nullptr);
    return ptr;
  }

  const std::shared_ptr<int64_t> ReducerProd::apply_int64(const int64_t* data, int64_t offset, const Index64& parents, int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength], util::array_deleter<int64_t>());
    struct Error err = awkward_reduce_prod_int64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, "'prod'", nullptr);
    return ptr;
  }

  const std::shared_ptr<uint64_t> ReducerProd::apply_uint64(const uint64_t* data, int64_t offset, const Index64& parents, int64_t outlength) const {
    std::shared_ptr<uint64_t> ptr(new uint64_t[(size_t)outlength], util::array_deleter<uint64_t>());
    struct Error err = awkward_reduce_prod_uint64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, "'prod'", nullptr);
    return ptr;
  }

  const std::shared_ptr<float> ReducerProd::apply_float32(const float* data, int64_t offset, const Index64& parents, int64_t outlength) const {
    std::shared_ptr<float> ptr(new float[(size_t)outlength], util::array_deleter<float>());
    struct Error err = awkward_reduce_prod_float32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, "'prod'", nullptr);
    return ptr;
  }

  const std::shared_ptr<double> ReducerProd::apply_float64(const double* data, int64_t offset, const Index64& parents, int64_t outlength) const {
    std::shared_ptr<double> ptr(new double[(size_t)outlength], util::array_deleter<double>());
    struct Error err = awkward_reduce_prod_float64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, "'prod'", nullptr);
    return ptr;
  }
}
