// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include <limits>

#include "awkward/cpu-kernels/reducers.h"

#include "awkward/Reducer.h"

namespace awkward {
  const std::string
  Reducer::return_type(const std::string& given_type) const {
    return given_type;
  }

  ssize_t
  Reducer::return_typesize(const std::string& given_type) const {
    if (given_type.compare("?") == 0) {
      return 1;
    }
    else if (given_type.compare("b") == 0) {
      return 1;
    }
    else if (given_type.compare("B") == 0  ||  given_type.compare("c") == 0) {
      return 1;
    }
    else if (given_type.compare("h") == 0) {
      return 2;
    }
    else if (given_type.compare("H") == 0) {
      return 2;
    }
#if defined _MSC_VER || defined __i386__
    else if (given_type.compare("l") == 0) {
#else
    else if (given_type.compare("i") == 0) {
#endif
      return 4;
    }
#if defined _MSC_VER || defined __i386__
      else if (given_type.compare("L") == 0) {
#else
    else if (given_type.compare("I") == 0) {
#endif
      return 4;
    }
#if defined _MSC_VER || defined __i386__
    else if (given_type.compare("q") == 0) {
#else
    else if (given_type.compare("l") == 0) {
#endif
      return 8;
    }
#if defined _MSC_VER || defined __i386__
    else if (given_type.compare("Q") == 0) {
#else
    else if (given_type.compare("L") == 0) {
#endif
      return 8;
    }
    else if (given_type.compare("f") == 0) {
      return 4;
    }
    else if (given_type.compare("d") == 0) {
      return 8;
    }
    else {
      throw std::runtime_error("this should be handled in NumpyArray");
    }
  }

  ////////// count

  const std::string
  ReducerCount::name() const {
    return "count";
  }

  const std::string
  ReducerCount::preferred_type() const {
    return "d";
  }

  ssize_t
  ReducerCount::preferred_typesize() const {
    return 8;
  }

  const std::string
  ReducerCount::return_type(const std::string& given_type) const {
#if defined _MSC_VER || defined __i386__
    return "q";
#else
    return "l";
#endif
  }

  ssize_t
  ReducerCount::return_typesize(const std::string& given_type) const {
    return 8;
  }

  const std::shared_ptr<void>
  ReducerCount::apply_bool(const bool* data,
                           int64_t offset,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength) const {
    // This is the only reducer that completely ignores the data.
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_count_64(
      ptr.get(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerCount::apply_int8(const int8_t* data,
                           int64_t offset,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength) const {
    return apply_bool(reinterpret_cast<const bool*>(data),
                      offset,
                      starts,
                      parents,
                      outlength);
  }

  const std::shared_ptr<void>
  ReducerCount::apply_uint8(const uint8_t* data,
                            int64_t offset,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength) const {
    return apply_bool(reinterpret_cast<const bool*>(data),
                      offset,
                      starts,
                      parents,
                      outlength);
  }

  const std::shared_ptr<void>
  ReducerCount::apply_int16(const int16_t* data,
                            int64_t offset,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength) const {
    return apply_bool(reinterpret_cast<const bool*>(data),
                      offset,
                      starts,
                      parents,
                      outlength);
  }

  const std::shared_ptr<void>
  ReducerCount::apply_uint16(const uint16_t* data,
                             int64_t offset,
                             const Index64& starts,
                             const Index64& parents,
                             int64_t outlength) const {
    return apply_bool(reinterpret_cast<const bool*>(data),
                      offset,
                      starts,
                      parents,
                      outlength);
  }

  const std::shared_ptr<void>
  ReducerCount::apply_int32(const int32_t* data,
                            int64_t offset,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength) const {
    return apply_bool(reinterpret_cast<const bool*>(data),
                      offset,
                      starts,
                      parents,
                      outlength);
  }

  const std::shared_ptr<void>
  ReducerCount::apply_uint32(const uint32_t* data,
                             int64_t offset,
                             const Index64& starts,
                             const Index64& parents,
                             int64_t outlength) const {
    return apply_bool(reinterpret_cast<const bool*>(data),
                      offset,
                      starts,
                      parents,
                      outlength);
  }

  const std::shared_ptr<void>
  ReducerCount::apply_int64(const int64_t* data,
                            int64_t offset,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength) const {
    return apply_bool(reinterpret_cast<const bool*>(data),
                      offset,
                      starts,
                      parents,
                      outlength);
  }

  const std::shared_ptr<void>
  ReducerCount::apply_uint64(const uint64_t* data,
                             int64_t offset,
                             const Index64& starts,
                             const Index64& parents,
                             int64_t outlength) const {
    return apply_bool(reinterpret_cast<const bool*>(data),
                      offset,
                      starts,
                      parents,
                      outlength);
  }

  const std::shared_ptr<void>
  ReducerCount::apply_float32(const float* data,
                              int64_t offset,
                              const Index64& starts,
                              const Index64& parents,
                              int64_t outlength) const {
    return apply_bool(reinterpret_cast<const bool*>(data),
                      offset,
                      starts,
                      parents,
                      outlength);
  }

  const std::shared_ptr<void>
  ReducerCount::apply_float64(const double* data,
                              int64_t offset,
                              const Index64& starts,
                              const Index64& parents,
                              int64_t outlength) const {
    return apply_bool(reinterpret_cast<const bool*>(data),
                      offset,
                      starts,
                      parents,
                      outlength);
  }

  ////////// count nonzero

  const std::string
  ReducerCountNonzero::name() const {
    return "count_nonzero";
  }

  const std::string
  ReducerCountNonzero::preferred_type() const {
    return "d";
  }

  ssize_t
  ReducerCountNonzero::preferred_typesize() const {
    return 8;
  }

  const std::string
  ReducerCountNonzero::return_type(const std::string& given_type) const {
#if defined _MSC_VER || defined __i386__
    return "q";
#else
    return "l";
#endif
  }

  ssize_t
  ReducerCountNonzero::return_typesize(const std::string& given_type) const {
    return 8;
  }

  const std::shared_ptr<void>
  ReducerCountNonzero::apply_bool(const bool* data,
                                  int64_t offset,
                                  const Index64& starts,
                                  const Index64& parents,
                                  int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_countnonzero_bool_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerCountNonzero::apply_int8(const int8_t* data,
                                  int64_t offset,
                                  const Index64& starts,
                                  const Index64& parents,
                                  int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_countnonzero_int8_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerCountNonzero::apply_uint8(const uint8_t* data,
                                   int64_t offset,
                                   const Index64& starts,
                                   const Index64& parents,
                                   int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_countnonzero_uint8_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerCountNonzero::apply_int16(const int16_t* data,
                                   int64_t offset,
                                   const Index64& starts,
                                   const Index64& parents,
                                   int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_countnonzero_int16_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerCountNonzero::apply_uint16(const uint16_t* data,
                                    int64_t offset,
                                    const Index64& starts,
                                    const Index64& parents,
                                    int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_countnonzero_uint16_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerCountNonzero::apply_int32(const int32_t* data,
                                   int64_t offset,
                                   const Index64& starts,
                                   const Index64& parents,
                                   int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_countnonzero_int32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerCountNonzero::apply_uint32(const uint32_t* data,
                                    int64_t offset,
                                    const Index64& starts,
                                    const Index64& parents,
                                    int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_countnonzero_uint32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerCountNonzero::apply_int64(const int64_t* data,
                                   int64_t offset,
                                   const Index64& starts,
                                   const Index64& parents,
                                   int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_countnonzero_int64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerCountNonzero::apply_uint64(const uint64_t* data,
                                    int64_t offset,
                                    const Index64& starts,
                                    const Index64& parents,
                                    int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_countnonzero_uint64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerCountNonzero::apply_float32(const float* data,
                                     int64_t offset,
                                     const Index64& starts,
                                     const Index64& parents,
                                     int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_countnonzero_float32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerCountNonzero::apply_float64(const double* data,
                                     int64_t offset,
                                     const Index64& starts,
                                     const Index64& parents,
                                     int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_countnonzero_float64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  ////////// sum (addition)

  const std::string
  ReducerSum::name() const {
    return "sum";
  }

  const std::string
  ReducerSum::preferred_type() const {
    return "d";
  }

  ssize_t
  ReducerSum::preferred_typesize() const {
    return 8;
  }

  const std::string
  ReducerSum::return_type(const std::string& given_type) const {
#if defined _MSC_VER || defined __i386__
    // if the array is 64-bit, even Windows and 32-bit platforms return 64-bit
    if (given_type.compare("q") == 0) {
      return "q";
    }
    if (given_type.compare("Q") == 0) {
      return "Q";
    }
#endif
    if (given_type.compare("?") == 0  ||
        given_type.compare("b") == 0  ||
        given_type.compare("h") == 0  ||
        given_type.compare("i") == 0  ||
        given_type.compare("l") == 0  ||
        given_type.compare("q") == 0) {
      // for _MSC_VER or __i386__, "l" means 32-bit
      // for MacOS/Linux 64-bit,   "l" means 64-bit
      return "l";
    }
    else if (
        given_type.compare("B") == 0  ||
        given_type.compare("H") == 0  ||
        given_type.compare("I") == 0  ||
        given_type.compare("L") == 0  ||
        given_type.compare("Q") == 0) {
      // for _MSC_VER or __i386__, "L" means unsigned 32-bit
      // for MacOS/Linux 64-bit,   "L" means unsigned 64-bit
      return "L";
    }
    else {
      return given_type;
    }
  }

  ssize_t
  ReducerSum::return_typesize(const std::string& given_type) const {
#if defined _MSC_VER || defined __i386__
    if (given_type.compare("q") == 0  ||
        given_type.compare("Q") == 0) {
      return 8;
    }
#endif
    if (given_type.compare("?") == 0  ||
        given_type.compare("b") == 0  ||
        given_type.compare("h") == 0  ||
        given_type.compare("i") == 0  ||
        given_type.compare("l") == 0  ||
        given_type.compare("q") == 0  ||
        given_type.compare("B") == 0  ||
        given_type.compare("H") == 0  ||
        given_type.compare("I") == 0  ||
        given_type.compare("L") == 0  ||
        given_type.compare("Q") == 0) {
#if defined _MSC_VER || defined __i386__
      return 4;
#else
      return 8;
#endif
    }
    else if (given_type.compare("f") == 0) {
      return 4;
    }
    else if (given_type.compare("d") == 0) {
      return 8;
    }
    else {
      throw std::runtime_error("this should be handled in NumpyArray");
    }
  }

  const std::shared_ptr<void>
  ReducerSum::apply_bool(const bool* data,
                         int64_t offset,
                         const Index64& starts,
                         const Index64& parents,
                         int64_t outlength) const {
#if defined _MSC_VER || defined __i386__
    std::shared_ptr<int32_t> ptr(new int32_t[(size_t)outlength],
                                 kernel::array_deleter<int32_t>());
    struct Error err = awkward_reduce_sum_int32_bool_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#else
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_sum_int64_bool_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#endif
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerSum::apply_int8(const int8_t* data,
                         int64_t offset,
                         const Index64& starts,
                         const Index64& parents,
                         int64_t outlength) const {
#if defined _MSC_VER || defined __i386__
    std::shared_ptr<int32_t> ptr(new int32_t[(size_t)outlength],
                                 kernel::array_deleter<int32_t>());
    struct Error err = awkward_reduce_sum_int32_int8_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#else
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_sum_int64_int8_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#endif
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerSum::apply_uint8(const uint8_t* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
#if defined _MSC_VER || defined __i386__
    std::shared_ptr<uint32_t> ptr(new uint32_t[(size_t)outlength],
                                  kernel::array_deleter<uint32_t>());
    struct Error err = awkward_reduce_sum_uint32_uint8_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#else
    std::shared_ptr<uint64_t> ptr(new uint64_t[(size_t)outlength],
                                  kernel::array_deleter<uint64_t>());
    struct Error err = awkward_reduce_sum_uint64_uint8_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#endif
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerSum::apply_int16(const int16_t* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
#if defined _MSC_VER || defined __i386__
    std::shared_ptr<int32_t> ptr(new int32_t[(size_t)outlength],
                                 kernel::array_deleter<int32_t>());
    struct Error err = awkward_reduce_sum_int32_int16_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#else
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_sum_int64_int16_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#endif
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerSum::apply_uint16(const uint16_t* data,
                           int64_t offset,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength) const {
#if defined _MSC_VER || defined __i386__
    std::shared_ptr<uint32_t> ptr(new uint32_t[(size_t)outlength],
                                  kernel::array_deleter<uint32_t>());
    struct Error err = awkward_reduce_sum_uint32_uint16_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#else
    std::shared_ptr<uint64_t> ptr(new uint64_t[(size_t)outlength],
                                  kernel::array_deleter<uint64_t>());
    struct Error err = awkward_reduce_sum_uint64_uint16_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#endif
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerSum::apply_int32(const int32_t* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
#if defined _MSC_VER || defined __i386__
    std::shared_ptr<int32_t> ptr(new int32_t[(size_t)outlength],
                                 kernel::array_deleter<int32_t>());
    struct Error err = awkward_reduce_sum_int32_int32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#else
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_sum_int64_int32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#endif
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerSum::apply_uint32(const uint32_t* data,
                           int64_t offset,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength) const {
#if defined _MSC_VER || defined __i386__
    std::shared_ptr<uint32_t> ptr(new uint32_t[(size_t)outlength],
                                  kernel::array_deleter<uint32_t>());
    struct Error err = awkward_reduce_sum_uint32_uint32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#else
    std::shared_ptr<uint64_t> ptr(new uint64_t[(size_t)outlength],
                                  kernel::array_deleter<uint64_t>());
    struct Error err = awkward_reduce_sum_uint64_uint32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#endif
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerSum::apply_int64(const int64_t* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_sum_int64_int64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerSum::apply_uint64(const uint64_t* data,
                           int64_t offset,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength) const {
    std::shared_ptr<uint64_t> ptr(new uint64_t[(size_t)outlength],
                                  kernel::array_deleter<uint64_t>());
    struct Error err = awkward_reduce_sum_uint64_uint64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerSum::apply_float32(const float* data,
                            int64_t offset,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength) const {
    std::shared_ptr<float> ptr(new float[(size_t)outlength],
                               kernel::array_deleter<float>());
    struct Error err = awkward_reduce_sum_float32_float32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerSum::apply_float64(const double* data,
                            int64_t offset,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength) const {
    std::shared_ptr<double> ptr(new double[(size_t)outlength],
                                kernel::array_deleter<double>());
    struct Error err = awkward_reduce_sum_float64_float64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  ////////// prod (multiplication)

  const std::string
  ReducerProd::name() const {
    return "prod";
  }

  const std::string
  ReducerProd::preferred_type() const {
#if defined _MSC_VER || defined __i386__
    return "q";
#else
    return "l";
#endif
  }

  ssize_t
  ReducerProd::preferred_typesize() const {
    return 8;
  }

  const std::string
  ReducerProd::return_type(const std::string& given_type) const {
#if defined _MSC_VER || defined __i386__
    // if the array is 64-bit, even Windows and 32-bit platforms return 64-bit
    if (given_type.compare("q") == 0) {
      return "q";
    }
    if (given_type.compare("Q") == 0) {
      return "Q";
    }
#endif
    if (given_type.compare("?") == 0  ||
        given_type.compare("b") == 0  ||
        given_type.compare("h") == 0  ||
        given_type.compare("i") == 0  ||
        given_type.compare("l") == 0  ||
        given_type.compare("q") == 0) {
      // for _MSC_VER or __i386__, "l" means 32-bit
      // for MacOS/Linux 64-bit,   "l" means 64-bit
      return "l";
    }
    else if (
        given_type.compare("B") == 0  ||
        given_type.compare("H") == 0  ||
        given_type.compare("I") == 0  ||
        given_type.compare("L") == 0  ||
        given_type.compare("Q") == 0) {
      // for _MSC_VER or __i386__, "L" means unsigned 32-bit
      // for MacOS/Linux 64-bit,   "L" means unsigned 64-bit
      return "L";
    }
    else {
      return given_type;
    }
  }

  ssize_t
  ReducerProd::return_typesize(const std::string& given_type) const {
#if defined _MSC_VER || defined __i386__
    if (given_type.compare("q") == 0  ||
        given_type.compare("Q") == 0) {
      return 8;
    }
#endif
    if (given_type.compare("?") == 0  ||
        given_type.compare("b") == 0  ||
        given_type.compare("h") == 0  ||
        given_type.compare("i") == 0  ||
        given_type.compare("l") == 0  ||
        given_type.compare("q") == 0  ||
        given_type.compare("B") == 0  ||
        given_type.compare("H") == 0  ||
        given_type.compare("I") == 0  ||
        given_type.compare("L") == 0  ||
        given_type.compare("Q") == 0) {
#if defined _MSC_VER || defined __i386__
      return 4;
#else
      return 8;
#endif
    }
    else if (given_type.compare("f") == 0) {
      return 4;
    }
    else if (given_type.compare("d") == 0) {
      return 8;
    }
    else {
      throw std::runtime_error("this should be handled in NumpyArray");
    }
  }

  const std::shared_ptr<void>
  ReducerProd::apply_bool(const bool* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
#if defined _MSC_VER || defined __i386__
    std::shared_ptr<int32_t> ptr(new int32_t[(size_t)outlength],
                                 kernel::array_deleter<int32_t>());
    struct Error err = awkward_reduce_prod_int32_bool_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#else
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_prod_int64_bool_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#endif
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerProd::apply_int8(const int8_t* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
#if defined _MSC_VER || defined __i386__
    std::shared_ptr<int32_t> ptr(new int32_t[(size_t)outlength],
                                 kernel::array_deleter<int32_t>());
    struct Error err = awkward_reduce_prod_int32_int8_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#else
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_prod_int64_int8_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#endif
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerProd::apply_uint8(const uint8_t* data,
                           int64_t offset,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength) const {
#if defined _MSC_VER || defined __i386__
    std::shared_ptr<uint32_t> ptr(new uint32_t[(size_t)outlength],
                                  kernel::array_deleter<uint32_t>());
    struct Error err = awkward_reduce_prod_uint32_uint8_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#else
    std::shared_ptr<uint64_t> ptr(new uint64_t[(size_t)outlength],
                                  kernel::array_deleter<uint64_t>());
    struct Error err = awkward_reduce_prod_uint64_uint8_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#endif
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerProd::apply_int16(const int16_t* data,
                           int64_t offset,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength) const {
#if defined _MSC_VER || defined __i386__
    std::shared_ptr<int32_t> ptr(new int32_t[(size_t)outlength],
                                 kernel::array_deleter<int32_t>());
    struct Error err = awkward_reduce_prod_int32_int16_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#else
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_prod_int64_int16_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#endif
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerProd::apply_uint16(const uint16_t* data,
                            int64_t offset,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength) const {
#if defined _MSC_VER || defined __i386__
    std::shared_ptr<uint32_t> ptr(new uint32_t[(size_t)outlength],
                                  kernel::array_deleter<uint32_t>());
    struct Error err = awkward_reduce_prod_uint32_uint16_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#else
    std::shared_ptr<uint64_t> ptr(new uint64_t[(size_t)outlength],
                                  kernel::array_deleter<uint64_t>());
    struct Error err = awkward_reduce_prod_uint64_uint16_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#endif
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerProd::apply_int32(const int32_t* data,
                           int64_t offset,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength) const {
#if defined _MSC_VER || defined __i386__
    std::shared_ptr<int32_t> ptr(new int32_t[(size_t)outlength],
                                 kernel::array_deleter<int32_t>());
    struct Error err = awkward_reduce_prod_int32_int32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#else
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_prod_int64_int32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#endif
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerProd::apply_uint32(const uint32_t* data,
                            int64_t offset,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength) const {
#if defined _MSC_VER || defined __i386__
    std::shared_ptr<uint32_t> ptr(new uint32_t[(size_t)outlength],
                                  kernel::array_deleter<uint32_t>());
    struct Error err = awkward_reduce_prod_uint32_uint32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#else
    std::shared_ptr<uint64_t> ptr(new uint64_t[(size_t)outlength],
                                  kernel::array_deleter<uint64_t>());
    struct Error err = awkward_reduce_prod_uint64_uint32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
#endif
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerProd::apply_int64(const int64_t* data,
                           int64_t offset,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_prod_int64_int64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerProd::apply_uint64(const uint64_t* data,
                            int64_t offset,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength) const {
    std::shared_ptr<uint64_t> ptr(new uint64_t[(size_t)outlength],
                                  kernel::array_deleter<uint64_t>());
    struct Error err = awkward_reduce_prod_uint64_uint64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerProd::apply_float32(const float* data,
                             int64_t offset,
                             const Index64& starts,
                             const Index64& parents,
                             int64_t outlength) const {
    std::shared_ptr<float> ptr(new float[(size_t)outlength],
                               kernel::array_deleter<float>());
    struct Error err = awkward_reduce_prod_float32_float32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerProd::apply_float64(const double* data,
                             int64_t offset,
                             const Index64& starts,
                             const Index64& parents,
                             int64_t outlength) const {
    std::shared_ptr<double> ptr(new double[(size_t)outlength],
                                kernel::array_deleter<double>());
    struct Error err = awkward_reduce_prod_float64_float64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  ////////// any (logical or)

  const std::string
  ReducerAny::name() const {
    return "any";
  }

  const std::string
  ReducerAny::preferred_type() const {
    return "?";
  }

  ssize_t
  ReducerAny::preferred_typesize() const {
    return 1;
  }

  const std::string
  ReducerAny::return_type(const std::string& given_type) const {
    return "?";
  }

  ssize_t
  ReducerAny::return_typesize(const std::string& given_type) const {
    return 1;
  }

  const std::shared_ptr<void>
  ReducerAny::apply_bool(const bool* data,
                         int64_t offset,
                         const Index64& starts,
                         const Index64& parents,
                         int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_sum_bool_bool_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerAny::apply_int8(const int8_t* data,
                         int64_t offset,
                         const Index64& starts,
                         const Index64& parents,
                         int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_sum_bool_int8_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerAny::apply_uint8(const uint8_t* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_sum_bool_uint8_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerAny::apply_int16(const int16_t* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_sum_bool_int16_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerAny::apply_uint16(const uint16_t* data,
                           int64_t offset,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_sum_bool_uint16_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerAny::apply_int32(const int32_t* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_sum_bool_int32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerAny::apply_uint32(const uint32_t* data,
                           int64_t offset,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_sum_bool_uint32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerAny::apply_int64(const int64_t* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_sum_bool_int64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerAny::apply_uint64(const uint64_t* data,
                           int64_t offset,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_sum_bool_uint64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerAny::apply_float32(const float* data,
                            int64_t offset,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_sum_bool_float32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerAny::apply_float64(const double* data,
                            int64_t offset,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_sum_bool_float64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  ////////// all (logical and)

  const std::string
  ReducerAll::name() const {
    return "all";
  }

  const std::string
  ReducerAll::preferred_type() const {
    return "?";
  }

  ssize_t
  ReducerAll::preferred_typesize() const {
    return 1;
  }

  const std::string
  ReducerAll::return_type(const std::string& given_type) const {
    return "?";
  }

  ssize_t
  ReducerAll::return_typesize(const std::string& given_type) const {
    return 1;
  }

  const std::shared_ptr<void>
  ReducerAll::apply_bool(const bool* data,
                         int64_t offset,
                         const Index64& starts,
                         const Index64& parents,
                         int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_prod_bool_bool_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerAll::apply_int8(const int8_t* data,
                         int64_t offset,
                         const Index64& starts,
                         const Index64& parents,
                         int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_prod_bool_int8_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerAll::apply_uint8(const uint8_t* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_prod_bool_uint8_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerAll::apply_int16(const int16_t* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_prod_bool_int16_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerAll::apply_uint16(const uint16_t* data,
                           int64_t offset,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_prod_bool_uint16_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerAll::apply_int32(const int32_t* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_prod_bool_int32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerAll::apply_uint32(const uint32_t* data,
                           int64_t offset,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_prod_bool_uint32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerAll::apply_int64(const int64_t* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_prod_bool_int64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerAll::apply_uint64(const uint64_t* data,
                           int64_t offset,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_prod_bool_uint64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerAll::apply_float32(const float* data,
                            int64_t offset,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_prod_bool_float32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerAll::apply_float64(const double* data,
                            int64_t offset,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_prod_bool_float64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  ////////// min (minimum, in which infinity is the identity)

  const std::string
  ReducerMin::name() const {
    return "min";
  }

  const std::string
  ReducerMin::preferred_type() const {
    return "d";
  }

  ssize_t
  ReducerMin::preferred_typesize() const {
    return 8;
  }

  const std::shared_ptr<void>
  ReducerMin::apply_bool(const bool* data,
                         int64_t offset,
                         const Index64& starts,
                         const Index64& parents,
                         int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_prod_bool_bool_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerMin::apply_int8(const int8_t* data,
                         int64_t offset,
                         const Index64& starts,
                         const Index64& parents,
                         int64_t outlength) const {
    std::shared_ptr<int8_t> ptr(new int8_t[(size_t)outlength],
                                kernel::array_deleter<int8_t>());
    struct Error err = awkward_reduce_min_int8_int8_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength,
      std::numeric_limits<int8_t>::max());
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerMin::apply_uint8(const uint8_t* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
    std::shared_ptr<uint8_t> ptr(new uint8_t[(size_t)outlength],
                                 kernel::array_deleter<uint8_t>());
    struct Error err = awkward_reduce_min_uint8_uint8_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength,
      std::numeric_limits<uint8_t>::max());
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerMin::apply_int16(const int16_t* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
    std::shared_ptr<int16_t> ptr(new int16_t[(size_t)outlength],
                                 kernel::array_deleter<int16_t>());
    struct Error err = awkward_reduce_min_int16_int16_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength,
      std::numeric_limits<int16_t>::max());
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerMin::apply_uint16(const uint16_t* data,
                           int64_t offset,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength) const {
    std::shared_ptr<uint16_t> ptr(new uint16_t[(size_t)outlength],
                                  kernel::array_deleter<uint16_t>());
    struct Error err = awkward_reduce_min_uint16_uint16_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength,
      std::numeric_limits<uint16_t>::max());
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerMin::apply_int32(const int32_t* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
    std::shared_ptr<int32_t> ptr(new int32_t[(size_t)outlength],
                                 kernel::array_deleter<int32_t>());
    struct Error err = awkward_reduce_min_int32_int32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength,
      std::numeric_limits<int32_t>::max());
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerMin::apply_uint32(const uint32_t* data,
                           int64_t offset,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength) const {
    std::shared_ptr<uint32_t> ptr(new uint32_t[(size_t)outlength],
                                  kernel::array_deleter<uint32_t>());
    struct Error err = awkward_reduce_min_uint32_uint32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength,
      std::numeric_limits<uint32_t>::max());
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerMin::apply_int64(const int64_t* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_min_int64_int64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength,
      std::numeric_limits<int64_t>::max());
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerMin::apply_uint64(const uint64_t* data,
                           int64_t offset,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength) const {
    std::shared_ptr<uint64_t> ptr(new uint64_t[(size_t)outlength],
                                  kernel::array_deleter<uint64_t>());
    struct Error err = awkward_reduce_min_uint64_uint64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength,
      std::numeric_limits<uint64_t>::max());
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerMin::apply_float32(const float* data,
                            int64_t offset,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength) const {
    std::shared_ptr<float> ptr(new float[(size_t)outlength],
                               kernel::array_deleter<float>());
    struct Error err = awkward_reduce_min_float32_float32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength,
      std::numeric_limits<float>::infinity());
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerMin::apply_float64(const double* data,
                            int64_t offset,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength) const {
    std::shared_ptr<double> ptr(new double[(size_t)outlength],
                                kernel::array_deleter<double>());
    struct Error err = awkward_reduce_min_float64_float64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength,
      std::numeric_limits<double>::infinity());
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  ////////// max (maximum, in which -infinity is the identity)

  const std::string
  ReducerMax::name() const {
    return "max";
  }

  const std::string
  ReducerMax::preferred_type() const {
    return "d";
  }

  ssize_t
  ReducerMax::preferred_typesize() const {
    return 8;
  }

  const std::shared_ptr<void>
  ReducerMax::apply_bool(const bool* data,
                         int64_t offset,
                         const Index64& starts,
                         const Index64& parents,
                         int64_t outlength) const {
    std::shared_ptr<bool> ptr(new bool[(size_t)outlength],
                              kernel::array_deleter<bool>());
    struct Error err = awkward_reduce_sum_bool_bool_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerMax::apply_int8(const int8_t* data,
                         int64_t offset,
                         const Index64& starts,
                         const Index64& parents,
                         int64_t outlength) const {
    std::shared_ptr<int8_t> ptr(new int8_t[(size_t)outlength],
                                kernel::array_deleter<int8_t>());
    struct Error err = awkward_reduce_max_int8_int8_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength,
      std::numeric_limits<int8_t>::min());
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerMax::apply_uint8(const uint8_t* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
    std::shared_ptr<uint8_t> ptr(new uint8_t[(size_t)outlength],
                                 kernel::array_deleter<uint8_t>());
    struct Error err = awkward_reduce_max_uint8_uint8_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength,
      std::numeric_limits<uint8_t>::min());
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerMax::apply_int16(const int16_t* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
    std::shared_ptr<int16_t> ptr(new int16_t[(size_t)outlength],
                                 kernel::array_deleter<int16_t>());
    struct Error err = awkward_reduce_max_int16_int16_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength,
      std::numeric_limits<int16_t>::min());
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerMax::apply_uint16(const uint16_t* data,
                           int64_t offset,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength) const {
    std::shared_ptr<uint16_t> ptr(new uint16_t[(size_t)outlength],
                                  kernel::array_deleter<uint16_t>());
    struct Error err = awkward_reduce_max_uint16_uint16_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength,
      std::numeric_limits<uint16_t>::min());
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerMax::apply_int32(const int32_t* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
    std::shared_ptr<int32_t> ptr(new int32_t[(size_t)outlength],
                                 kernel::array_deleter<int32_t>());
    struct Error err = awkward_reduce_max_int32_int32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength,
      std::numeric_limits<int32_t>::min());
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerMax::apply_uint32(const uint32_t* data,
                           int64_t offset,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength) const {
    std::shared_ptr<uint32_t> ptr(new uint32_t[(size_t)outlength],
                                  kernel::array_deleter<uint32_t>());
    struct Error err = awkward_reduce_max_uint32_uint32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength,
      std::numeric_limits<uint32_t>::min());
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerMax::apply_int64(const int64_t* data,
                          int64_t offset,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_max_int64_int64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength,
      std::numeric_limits<int64_t>::min());
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerMax::apply_uint64(const uint64_t* data,
                           int64_t offset,
                           const Index64& starts,
                           const Index64& parents,
                           int64_t outlength) const {
    std::shared_ptr<uint64_t> ptr(new uint64_t[(size_t)outlength],
                                  kernel::array_deleter<uint64_t>());
    struct Error err = awkward_reduce_max_uint64_uint64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength,
      std::numeric_limits<uint64_t>::min());
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerMax::apply_float32(const float* data,
                            int64_t offset,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength) const {
    std::shared_ptr<float> ptr(new float[(size_t)outlength],
                               kernel::array_deleter<float>());
    struct Error err = awkward_reduce_max_float32_float32_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength,
      -std::numeric_limits<float>::infinity());
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerMax::apply_float64(const double* data,
                            int64_t offset,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength) const {
    std::shared_ptr<double> ptr(new double[(size_t)outlength],
                                kernel::array_deleter<double>());
    struct Error err = awkward_reduce_max_float64_float64_64(
      ptr.get(),
      data,
      offset,
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength,
      -std::numeric_limits<double>::infinity());
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  ////////// argmin (argument minimum, in which -1 is the identity)

  const std::string
  ReducerArgmin::name() const {
    return "argmin";
  }

  const std::string
  ReducerArgmin::preferred_type() const {
#if defined _MSC_VER || defined __i386__
    return "q";
#else
    return "l";
#endif
  }

  ssize_t
  ReducerArgmin::preferred_typesize() const {
    return 8;
  }

  const std::string
  ReducerArgmin::return_type(const std::string& given_type) const {
#if defined _MSC_VER || defined __i386__
    return "q";
#else
    return "l";
#endif
  }

  ssize_t
  ReducerArgmin::return_typesize(const std::string& given_type) const {
    return 8;
  }

  const std::shared_ptr<void>
  ReducerArgmin::apply_bool(const bool* data,
                            int64_t offset,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmin_bool_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerArgmin::apply_int8(const int8_t* data,
                            int64_t offset,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmin_int8_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerArgmin::apply_uint8(const uint8_t* data,
                             int64_t offset,
                             const Index64& starts,
                             const Index64& parents,
                             int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmin_uint8_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerArgmin::apply_int16(const int16_t* data,
                             int64_t offset,
                             const Index64& starts,
                             const Index64& parents,
                             int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmin_int16_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerArgmin::apply_uint16(const uint16_t* data,
                              int64_t offset,
                              const Index64& starts,
                              const Index64& parents,
                              int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmin_uint16_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerArgmin::apply_int32(const int32_t* data,
                             int64_t offset,
                             const Index64& starts,
                             const Index64& parents,
                             int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmin_int32_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerArgmin::apply_uint32(const uint32_t* data,
                              int64_t offset,
                              const Index64& starts,
                              const Index64& parents,
                              int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmin_uint32_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerArgmin::apply_int64(const int64_t* data,
                             int64_t offset,
                             const Index64& starts,
                             const Index64& parents,
                             int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmin_int64_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerArgmin::apply_uint64(const uint64_t* data,
                              int64_t offset,
                              const Index64& starts,
                              const Index64& parents,
                              int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmin_uint64_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerArgmin::apply_float32(const float* data,
                               int64_t offset,
                               const Index64& starts,
                               const Index64& parents,
                               int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmin_float32_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerArgmin::apply_float64(const double* data,
                               int64_t offset,
                               const Index64& starts,
                               const Index64& parents,
                               int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmin_float64_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  ////////// argmax (argument maximum, in which -1 is the identity)

  const std::string
  ReducerArgmax::name() const {
    return "argmax";
  }

  const std::string
  ReducerArgmax::preferred_type() const {
#if defined _MSC_VER || defined __i386__
    return "q";
#else
    return "l";
#endif
  }

  ssize_t
  ReducerArgmax::preferred_typesize() const {
    return 8;
  }

  const std::string
  ReducerArgmax::return_type(const std::string& given_type) const {
#if defined _MSC_VER || defined __i386__
    return "q";
#else
    return "l";
#endif
  }

  ssize_t
  ReducerArgmax::return_typesize(const std::string& given_type) const {
    return 8;
  }

  const std::shared_ptr<void>
  ReducerArgmax::apply_bool(const bool* data,
                            int64_t offset,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmax_bool_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerArgmax::apply_int8(const int8_t* data,
                            int64_t offset,
                            const Index64& starts,
                            const Index64& parents,
                            int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmax_int8_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerArgmax::apply_uint8(const uint8_t* data,
                             int64_t offset,
                             const Index64& starts,
                             const Index64& parents,
                             int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmax_uint8_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerArgmax::apply_int16(const int16_t* data,
                             int64_t offset,
                             const Index64& starts,
                             const Index64& parents,
                             int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmax_int16_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerArgmax::apply_uint16(const uint16_t* data,
                              int64_t offset,
                              const Index64& starts,
                              const Index64& parents,
                              int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmax_uint16_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerArgmax::apply_int32(const int32_t* data,
                             int64_t offset,
                             const Index64& starts,
                             const Index64& parents,
                             int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmax_int32_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerArgmax::apply_uint32(const uint32_t* data,
                              int64_t offset,
                              const Index64& starts,
                              const Index64& parents,
                              int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmax_uint32_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerArgmax::apply_int64(const int64_t* data,
                             int64_t offset,
                             const Index64& starts,
                             const Index64& parents,
                             int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmax_int64_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerArgmax::apply_uint64(const uint64_t* data,
                              int64_t offset,
                              const Index64& starts,
                              const Index64& parents,
                              int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmax_uint64_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerArgmax::apply_float32(const float* data,
                               int64_t offset,
                               const Index64& starts,
                               const Index64& parents,
                               int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmax_float32_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

  const std::shared_ptr<void>
  ReducerArgmax::apply_float64(const double* data,
                               int64_t offset,
                               const Index64& starts,
                               const Index64& parents,
                               int64_t outlength) const {
    std::shared_ptr<int64_t> ptr(new int64_t[(size_t)outlength],
                                
                                 kernel::array_deleter<int64_t>());
    struct Error err = awkward_reduce_argmax_float64_64(
      ptr.get(),
      data,
      offset,
      starts.ptr().get(),
      starts.offset(),
      parents.ptr().get(),
      parents.offset(),
      parents.length(),
      outlength);
    util::handle_error(err, util::quote(name(), true), nullptr);
    return ptr;
  }

}
