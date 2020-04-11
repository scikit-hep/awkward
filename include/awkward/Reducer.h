// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_REDUCER_H_
#define AWKWARD_REDUCER_H_

#include <memory>

#include "awkward/Index.h"

namespace awkward {
  /// @class Reducer
  ///
  /// @brief Abstract class for all reducer algorithms.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL Reducer {
  public:
    /// @brief Name of the reducer algorithm.
    virtual const std::string
      name() const = 0;

    /// @brief Data type to prefer, as a pybind11 format string, if the array
    /// has UnknownType.
    virtual const std::string
      preferred_type() const = 0;

    /// @brief Number of bytes in the data type to prefer if the array has
    /// UnknownType.
    virtual ssize_t
      preferred_typesize() const = 0;

    /// @brief Return type for a `given_type` as a pybind11 format string.
    virtual const std::string
      return_type(const std::string& given_type) const;

    /// @brief Number of bytes in the return type for a `given_type`.
    virtual ssize_t
      return_typesize(const std::string& given_type) const;

    /// @brief Apply the reducer algorithm to an array of boolean values.
    ///
    /// @param data The array to reduce.
    /// @param offset The location of the first item in the array.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_bool(const bool* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const = 0;

    /// @brief Apply the reducer algorithm to an array of signed 8-bit
    /// integer values.
    ///
    /// @param data The array to reduce.
    /// @param offset The location of the first item in the array.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const = 0;

    /// @brief Apply the reducer algorithm to an array of unsigned 8-bit
    /// integer values.
    ///
    /// @param data The array to reduce.
    /// @param offset The location of the first item in the array.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const = 0;

    /// @brief Apply the reducer algorithm to an array of signed 16-bit
    /// integer values.
    ///
    /// @param data The array to reduce.
    /// @param offset The location of the first item in the array.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const = 0;

    /// @brief Apply the reducer algorithm to an array of unsigned 16-bit
    /// integer values.
    ///
    /// @param data The array to reduce.
    /// @param offset The location of the first item in the array.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const = 0;

    /// @brief Apply the reducer algorithm to an array of signed 32-bit
    /// integer values.
    ///
    /// @param data The array to reduce.
    /// @param offset The location of the first item in the array.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const = 0;

    /// @brief Apply the reducer algorithm to an array of unsigned 32-bit
    /// integer values.
    ///
    /// @param data The array to reduce.
    /// @param offset The location of the first item in the array.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const = 0;

    /// @brief Apply the reducer algorithm to an array of signed 64-bit
    /// integer values.
    ///
    /// @param data The array to reduce.
    /// @param offset The location of the first item in the array.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const = 0;

    /// @brief Apply the reducer algorithm to an array of unsigned 64-bit
    /// integer values.
    ///
    /// @param data The array to reduce.
    /// @param offset The location of the first item in the array.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const = 0;

    /// @brief Apply the reducer algorithm to an array of 32-bit
    /// floating-point values.
    ///
    /// @param data The array to reduce.
    /// @param offset The location of the first item in the array.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_float32(const float* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const = 0;

    /// @brief Apply the reducer algorithm to an array of 64-bit
    /// floating-point values.
    ///
    /// @param data The array to reduce.
    /// @param offset The location of the first item in the array.
    /// @param starts An integer array indicating where each group to combine
    /// starts.
    /// @param parents An integer array indicating which group each element
    /// belongs to.
    /// @param outlength The length of the output array (equal to the number
    /// of groups).
    virtual const std::shared_ptr<void>
      apply_float64(const double* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const = 0;
  };

  /// @class ReducerCount
  ///
  /// @brief Reducer algorithm that simply counts items. The identity is `0`.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL EXPORT_TYPE ReducerCount: public Reducer {
  public:
    /// @brief Name of the reducer algorithm: `"count"`.
    const std::string
      name() const override;

    /// @copydoc Reducer::preferred_type()
    ///
    /// The preferred type for ReducerCount is `double`: `"d"`, 8 bytes.
    const std::string
      preferred_type() const override;

    /// @copydoc Reducer::preferred_typesize()
    ///
    /// The preferred type for ReducerCount is `double`: `"d"`, 8 bytes.
    ssize_t
      preferred_typesize() const override;

    /// @copydoc Reducer::return_type()
    ///
    /// The return type for ReducerCount is `int64`: `"q"` (32-bit systems
    /// or Windows) or `"l"` (other systems), 8 bytes.
    const std::string
      return_type(const std::string& given_type) const override;

    /// @copydoc Reducer::return_typesize()
    ///
    /// The return type for ReducerCount is `int64`: `"q"` (32-bit systems
    /// or Windows) or `"l"` (other systems), 8 bytes.
    ssize_t
      return_typesize(const std::string& given_type) const override;

    const std::shared_ptr<void>
      apply_bool(const bool* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float32(const float* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float64(const double* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;
  };

  /// @class ReducerCountNonzero
  ///
  /// @brief Reducer algorithm that counts non-zero items. The identity is `0`.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL EXPORT_TYPE ReducerCountNonzero: public Reducer {
  public:
    /// @brief Name of the reducer algorithm: `"count_nonzero"`.
    const std::string
      name() const override;

    /// @copydoc Reducer::preferred_type()
    ///
    /// The preferred type for ReducerCountNonzero is `double`: `"d"`, 8
    /// bytes.
    const std::string
      preferred_type() const override;

    /// @copydoc Reducer::preferred_typesize()
    ///
    /// The preferred type for ReducerCountNonzero is `double`: `"d"`, 8
    /// bytes.
    ssize_t
      preferred_typesize() const override;

    /// @copydoc Reducer::return_type()
    ///
    /// The return type for ReducerCountNonzero is `int64`: `"q"` (32-bit
    /// systems or Windows) or `"l"` (other systems), 8 bytes.
    const std::string
      return_type(const std::string& given_type) const override;

    /// @copydoc Reducer::return_typesize()
    ///
    /// The return type for ReducerCountNonzero is `int64`: `"q"` (32-bit
    /// systems or Windows) or `"l"` (other systems), 8 bytes.
    ssize_t
      return_typesize(const std::string& given_type) const override;

    const std::shared_ptr<void>
      apply_bool(const bool* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float32(const float* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float64(const double* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;
  };

  /// @class ReducerSum
  ///
  /// @brief Reducer algorithm that adds up items. The identity is `0`.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL EXPORT_TYPE ReducerSum: public Reducer {
  public:
    /// @brief Name of the reducer algorithm: `"sum"`.
    const std::string
      name() const override;

    /// @copydoc Reducer::preferred_type()
    ///
    /// The preferred type for ReducerSum is `double`: `"d"`, 8 bytes.
    const std::string
      preferred_type() const override;

    /// @copydoc Reducer::preferred_typesize()
    ///
    /// The preferred type for ReducerSum is `double`: `"d"`, 8 bytes.
    ssize_t
      preferred_typesize() const override;

    /// @copydoc Reducer::return_type()
    ///
    /// The return type for ReducerSum promotes integers and booleans to
    /// 64-bit but leaves floating-point number types as they are.
    const std::string
      return_type(const std::string& given_type) const override;

    /// @copydoc Reducer::return_typesize()
    ///
    /// The return type for ReducerSum promotes integers and booleans to
    /// 64-bit but leaves floating-point number types as they are.
    ssize_t
      return_typesize(const std::string& given_type) const override;

    const std::shared_ptr<void>
      apply_bool(const bool* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float32(const float* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float64(const double* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;
  };

  /// @class ReducerProd
  ///
  /// @brief Reducer algorithm that multiplies items. The identity is `1`.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL EXPORT_TYPE ReducerProd: public Reducer {
  public:
    /// @brief Name of the reducer algorithm: `"prod"`.
    const std::string
      name() const override;

    /// @copydoc Reducer::preferred_type()
    ///
    /// The preferred type for ReducerProd is `int64`: `"q"` (32-bit systems
    /// or Windows) or `"l"` (other systems), 8 bytes.
    const std::string
      preferred_type() const override;

    /// @copydoc Reducer::preferred_typesize()
    ///
    /// The preferred type for ReducerProd is `int64`: `"q"` (32-bit systems
    /// or Windows) or `"l"` (other systems), 8 bytes.
    ssize_t
      preferred_typesize() const override;

    /// @copydoc Reducer::return_type()
    ///
    /// The return type for ReducerProd promotes integers and booleans to
    /// 64-bit but leaves floating-point number types as they are.
    const std::string
      return_type(const std::string& given_type) const override;

    /// @copydoc Reducer::return_typesize()
    ///
    /// The return type for ReducerProd promotes integers and booleans to
    /// 64-bit but leaves floating-point number types as they are.
    ssize_t
      return_typesize(const std::string& given_type) const override;

    const std::shared_ptr<void>
      apply_bool(const bool* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float32(const float* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float64(const double* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;
  };

  /// @class ReducerAny
  ///
  /// @brief Reducer algorithm that returns `true` if any values are `true`,
  /// `false` otherwise. The identity is `false`.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL EXPORT_TYPE ReducerAny: public Reducer {
  public:
    /// @brief Name of the reducer algorithm: `"any"`.
    const std::string
      name() const override;

    /// @copydoc Reducer::preferred_type()
    ///
    /// The preferred type for ReducerAny is `boolean`: `"?"`, 1 byte.
    const std::string
      preferred_type() const override;

    /// @copydoc Reducer::preferred_typesize()
    ///
    /// The preferred type for ReducerAny is `boolean`: `"?"`, 1 byte.
    ssize_t
      preferred_typesize() const override;

    /// @copydoc Reducer::return_type()
    ///
    /// The return type for ReducerAny is `boolean`: `"?"`, 1 byte.
    const std::string
      return_type(const std::string& given_type) const override;

    /// @copydoc Reducer::return_typesize()
    ///
    /// The return type for ReducerAny is `boolean`: `"?"`, 1 byte.
    ssize_t
      return_typesize(const std::string& given_type) const override;

    const std::shared_ptr<void>
      apply_bool(const bool* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float32(const float* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float64(const double* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;
  };

  /// @class ReducerAll
  ///
  /// @brief Reducer algorithm that returns `true` if all values are `true`,
  /// `false` otherwise. The identity is `true`.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL EXPORT_TYPE ReducerAll: public Reducer {
  public:
    /// @brief Name of the reducer algorithm: `"all"`.
    const std::string
      name() const override;

    /// @copydoc Reducer::preferred_type()
    ///
    /// The preferred type for ReducerAll is `boolean`: `"?"`, 1 byte.
    const std::string
      preferred_type() const override;

    /// @copydoc Reducer::preferred_typesize()
    ///
    /// The preferred type for ReducerAll is `boolean`: `"?"`, 1 byte.
    ssize_t
      preferred_typesize() const override;

    /// @copydoc Reducer::return_type()
    ///
    /// The return type for ReducerAll is `boolean`: `"?"`, 1 byte.
    const std::string
      return_type(const std::string& given_type) const override;

    /// @copydoc Reducer::return_typesize()
    ///
    /// The return type for ReducerAll is `boolean`: `"?"`, 1 byte.
    ssize_t
      return_typesize(const std::string& given_type) const override;

    const std::shared_ptr<void>
      apply_bool(const bool* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float32(const float* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float64(const double* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;
  };

  /// @class ReducerMin
  ///
  /// @brief Reducer algorithm that returns the minimumm value. The identity
  /// is infinity or the largest possible value.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL EXPORT_TYPE ReducerMin: public Reducer {
  public:
    /// @brief Name of the reducer algorithm: `"min"`.
    const std::string
      name() const override;

    /// @copydoc Reducer::preferred_type()
    ///
    /// The preferred type for ReducerMin is `double`: `"d"`, 8 bytes.
    const std::string
      preferred_type() const override;

    /// @copydoc Reducer::preferred_typesize()
    ///
    /// The preferred type for ReducerMin is `double`: `"d"`, 8 bytes.
    ssize_t
      preferred_typesize() const override;

    const std::shared_ptr<void>
      apply_bool(const bool* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float32(const float* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float64(const double* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;
  };

  /// @class ReducerMax
  ///
  /// @brief Reducer algorithm that returns the maximum value. The identity
  /// is minus infinity or the smallest possible value.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL EXPORT_TYPE ReducerMax: public Reducer {
  public:
    /// @brief Name of the reducer algorithm: `"max"`.
    const std::string
      name() const override;

    /// @copydoc Reducer::preferred_type()
    ///
    /// The preferred type for ReducerMax is `double`: `"d"`, 8 bytes.
    const std::string
      preferred_type() const override;

    /// @copydoc Reducer::preferred_typesize()
    ///
    /// The preferred type for ReducerMax is `double`: `"d"`, 8 bytes.
    ssize_t
      preferred_typesize() const override;

    const std::shared_ptr<void>
      apply_bool(const bool* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float32(const float* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float64(const double* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;
  };

  /// @class ReducerArgmin
  ///
  /// @brief Reducer algorithm that returns the position of the minimum value.
  /// The identity is meaningless and should be covered using `mask = true`.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL EXPORT_TYPE ReducerArgmin: public Reducer {
  public:
    /// @brief Name of the reducer algorithm: `"argmin"`.
    const std::string
      name() const override;

    /// @copydoc Reducer::preferred_type()
    ///
    /// The preferred type for ReducerArgmin is `int64`: `"q"` (32-bit systems
    /// or Windows) or `"l"` (other systems), 8 bytes.
    const std::string
      preferred_type() const override;

    /// @copydoc Reducer::preferred_typesize()
    ///
    /// The preferred type for ReducerArgmin is `int64`: `"q"` (32-bit systems
    /// or Windows) or `"l"` (other systems), 8 bytes.
    ssize_t
      preferred_typesize() const override;

    /// @copydoc Reducer::return_type()
    ///
    /// The return type for ReducerArgmin is `int64`: `"q"` (32-bit systems
    /// or Windows) or `"l"` (other systems), 8 bytes.
    const std::string
      return_type(const std::string& given_type) const override;

    /// @copydoc Reducer::return_typesize()
    ///
    /// The return type for ReducerArgmin is `int64`: `"q"` (32-bit systems
    /// or Windows) or `"l"` (other systems), 8 bytes.
    ssize_t
      return_typesize(const std::string& given_type) const override;

    const std::shared_ptr<void>
      apply_bool(const bool* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float32(const float* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float64(const double* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;
  };

  /// @class ReducerArgmax
  ///
  /// @brief Reducer algorithm that returns the position of the maximum value.
  /// The identity is meaningless and should be covered using `mask = true`.
  ///
  /// Reducers have no parameters or state. They are classes for convenience,
  /// to separate {@link Content#reduce_next Content::reduce_next}, determining
  /// which values to combine, from the choice of reducer algorithm.
  class EXPORT_SYMBOL EXPORT_TYPE ReducerArgmax: public Reducer {
  public:
    /// @brief Name of the reducer algorithm: `"argmax"`.
    const std::string
      name() const override;

    /// @copydoc Reducer::preferred_type()
    ///
    /// The preferred type for ReducerArgmax is `int64`: `"q"` (32-bit systems
    /// or Windows) or `"l"` (other systems), 8 bytes.
    const std::string
      preferred_type() const override;

    /// @copydoc Reducer::preferred_typesize()
    ///
    /// The preferred type for ReducerArgmax is `int64`: `"q"` (32-bit systems
    /// or Windows) or `"l"` (other systems), 8 bytes.
    ssize_t
      preferred_typesize() const override;

    /// @copydoc Reducer::return_type()
    ///
    /// The return type for ReducerArgmax is `int64`: `"q"` (32-bit systems
    /// or Windows) or `"l"` (other systems), 8 bytes.
    const std::string
      return_type(const std::string& given_type) const override;

    /// @copydoc Reducer::return_typesize()
    ///
    /// The return type for ReducerArgmax is `int64`: `"q"` (32-bit systems
    /// or Windows) or `"l"` (other systems), 8 bytes.
    ssize_t
      return_typesize(const std::string& given_type) const override;

    const std::shared_ptr<void>
      apply_bool(const bool* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int8(const int8_t* data,
                 int64_t offset,
                 const Index64& starts,
                 const Index64& parents,
                 int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint8(const uint8_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int16(const int16_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint16(const uint16_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int32(const int32_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint32(const uint32_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_int64(const int64_t* data,
                  int64_t offset,
                  const Index64& starts,
                  const Index64& parents,
                  int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_uint64(const uint64_t* data,
                   int64_t offset,
                   const Index64& starts,
                   const Index64& parents,
                   int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float32(const float* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;

    const std::shared_ptr<void>
      apply_float64(const double* data,
                    int64_t offset,
                    const Index64& starts,
                    const Index64& parents,
                    int64_t outlength) const override;
  };

}

#endif // AWKWARD_REDUCER_H_
