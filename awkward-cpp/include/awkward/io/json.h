// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARD_IO_JSON_H_
#define AWKWARD_IO_JSON_H_

#include <complex>
#include <cstdio>
#include <string>

#include "awkward/common.h"
#include "awkward/builder/Builder.h"
#include "awkward/builder/ArrayBuilder.h"
#include "awkward/BuilderOptions.h"
#include "awkward/GrowableBuffer.h"
#include "awkward/util.h"

namespace awkward {
  /// @class FileLikeObject
  ///
  /// @brief Abstract class to represent a file-like object, something with
  /// a `read(num_bytes)` method. Satisfies RapidJSON's Stream interface.
  class FileLikeObject {
  public:
    virtual int64_t read(int64_t num_bytes, char* buffer) = 0;
  };

  /// @brief Parses a JSON-encoded file-like object using an
  /// ArrayBuilder.
  ///
  /// @param source File-like object wrapped with the FileLikeObject
  /// abstraction (borrowed reference).
  /// @param builder To build the array.
  /// @param buffersize Number of bytes for an intermediate buffer.
  /// @param read_one If true, read only one JSON object (with an error if
  /// there's more); otherwise, read a stream of concatenated objects (may
  /// be separated by newlines, but we don't check).
  /// @param nan_string User-defined string for a not-a-number (NaN) value
  /// representation in JSON format.
  /// @param infinity_string User-defined string for a positive infinity
  /// representation in JSON format.
  /// @param minus_infinity_string User-defined string for a negative
  /// infinity representation in JSON format.
  EXPORT_SYMBOL void
    fromjsonobject(FileLikeObject* source,
                   ArrayBuilder& builder,
                   int64_t buffersize,
                   bool read_one,
                   const char* nan_string = nullptr,
                   const char* posinf_string = nullptr,
                   const char* neginf_string = nullptr);

  class EXPORT_SYMBOL FromJsonObjectSchema {
  public:
    FromJsonObjectSchema(FileLikeObject* source,
             int64_t buffersize,
             bool read_one,
             const char* nan_string,
             const char* posinf_string,
             const char* neginf_string,
             const char* jsonassembly,
             int64_t initial,
             double resize);

    // Delete copy constructor
    FromJsonObjectSchema(const FromJsonObjectSchema&) = delete;

    // Delete copy-assignment constructor
    FromJsonObjectSchema& operator=(FromJsonObjectSchema&) = delete;

    /// @brief HERE
    inline int64_t current_stack_depth() const noexcept {
      return current_stack_depth_;
    }

    /// @brief HERE
    inline int64_t current_instruction() const noexcept {
      return current_instruction_;
    }

    /// @brief HERE
    inline int64_t instruction() const noexcept {
      return instructions_.data()[current_instruction_ * 4];
    }

    /// @brief HERE
    inline int64_t argument1() const noexcept {
      return instructions_.data()[current_instruction_ * 4 + 1];
    }

    /// @brief HERE
    inline int64_t argument2() const noexcept {
      return instructions_.data()[current_instruction_ * 4 + 2];
    }

    /// @brief HERE
    inline int64_t argument3() const noexcept {
      return instructions_.data()[current_instruction_ * 4 + 3];
    }

    /// @brief HERE
    inline void step_forward() noexcept {
      current_instruction_++;
    }

    /// @brief HERE
    inline void step_backward() noexcept {
      current_instruction_--;
    }

    /// @brief HERE
    inline void push_stack(int64_t jump_to) noexcept {
      instruction_stack_.data()[current_stack_depth_] = current_instruction_;
      current_stack_depth_++;
      current_instruction_ = jump_to;
    }

    /// @brief HERE
    inline void pop_stack() noexcept {
      current_stack_depth_--;
      current_instruction_ = instruction_stack_.data()[current_stack_depth_];
    }

    /// @brief HERE
    inline int64_t find_enum(const char* str) noexcept {
      int64_t* offsets = string_offsets_.data();
      char* chars = characters_.data();
      int64_t stringsstart = argument2();
      int64_t start;
      int64_t stop;
      for (int64_t i = stringsstart;  i < argument3();  i++) {
        start = offsets[i];
        stop = offsets[i + 1];
        if (strncmp(str, &chars[start], (size_t)(stop - start)) == 0) {
          return i - stringsstart;
        }
      }
      return -1;
    }

    /// @brief HERE
    inline int64_t find_key(const char* str) noexcept {
      int64_t* offsets = string_offsets_.data();
      char* chars = characters_.data();
      int64_t i;
      int64_t j;
      int64_t stringi;
      int64_t start;
      int64_t stop;
      uint64_t chunkmask;
      // optimistic: fields in data are in the order specified by the schema
      if (argument1() != 0) {
        // increment the current (last seen) field with wrap-around
        record_current_field_[argument2()]++;
        if (record_current_field_[argument2()] == argument1()) {
          record_current_field_[argument2()] = 0;
        }
        j = record_current_field_[argument2()];
        // use the record_current_field_ (as j)
        i = current_instruction_ + 1 + j;
        stringi = instructions_.data()[i * 4 + 1];
        start = offsets[stringi];
        stop = offsets[stringi + 1];
        if (strncmp(str, &chars[start], (size_t)(stop - start)) == 0) {
          // ensure that the checklist bit is 1
          chunkmask = (uint64_t)1 << (j & 0x3f);
          if ((record_checklist_[argument2()][j >> 6] & chunkmask) == 0) {
            return -1;  // ignore the value of a duplicate key
          }
          // set the checklist bit to 0
          record_checklist_[argument2()][j >> 6] &= ~chunkmask;
          return key_instruction_at(i);
        }
      }
      // pessimistic: try all field names, starting from the first
      for (i = current_instruction_ + 1;  i <= current_instruction_ + argument1();  i++) {
        // not including the one optimistic trial
        if (i != current_instruction_ + 1 + record_current_field_[argument2()]) {
          stringi = instructions_.data()[i * 4 + 1];
          start = offsets[stringi];
          stop = offsets[stringi + 1];
          if (strncmp(str, &chars[start], (size_t)(stop - start)) == 0) {
            // set the record_current_field_
            j = i - (current_instruction_ + 1);
            record_current_field_[argument2()] = j;
            // ensure that the checklist bit is 1
            chunkmask = (uint64_t)1 << (j & 0x3f);
            if ((record_checklist_[argument2()][j >> 6] & chunkmask) == 0) {
              return -1;  // ignore the value of a duplicate key
            }
            // set the checklist bit to 0
            record_checklist_[argument2()][j >> 6] &= ~chunkmask;
            return key_instruction_at(i);
          }
        }
      }
      return -1;
    }

    /// @brief HERE
    inline bool key_already_filled(int64_t record_identifier, int64_t j) const noexcept {
      uint64_t chunkmask = (uint64_t)1 << (j & 0x3f);
      return (record_checklist_[record_identifier][j >> 6] & chunkmask) == 0;
    }

    /// @brief HERE
    inline int64_t key_instruction_at(int64_t i) const noexcept {
      return instructions_.data()[i * 4 + 2];
    }

    /// @brief HERE
    inline void start_object(int64_t keytableheader_instruction) noexcept {
      int64_t record_identifier = instructions_.data()[keytableheader_instruction * 4 + 2];
      record_checklist_[record_identifier].assign(
          record_checklist_init_[record_identifier].begin(),
          record_checklist_init_[record_identifier].end()
      );
    }

    /// @brief HERE
    inline bool end_object(int64_t keytableheader_instruction) const noexcept {
      int64_t record_identifier = instructions_.data()[keytableheader_instruction * 4 + 2];
      uint64_t should_be_zero = 0;
      for (uint64_t chunk : record_checklist_[record_identifier]) {
        should_be_zero |= chunk;
      }
      return should_be_zero == 0;
    }

    /// @brief HERE
    inline void write_int8(int64_t index, int8_t x) noexcept {
      buffers_uint8_[(size_t)index].append(*reinterpret_cast<uint8_t*>(&x));
    }

    /// @brief HERE
    inline void write_uint8(int64_t index, uint8_t x) noexcept {
      buffers_uint8_[(size_t)index].append(x);
    }

    /// @brief HERE
    inline void write_many_uint8(int64_t index, int64_t num_items, const uint8_t* values) noexcept {
      buffers_uint8_[(size_t)index].extend(values, (size_t)num_items);
    }

    /// @brief HERE
    inline void write_int64(int64_t index, int64_t x) noexcept {
      buffers_int64_[(size_t)index].append(x);
    }

    /// @brief HERE
    inline void write_uint64(int64_t index, uint64_t x) noexcept {
      buffers_int64_[(size_t)index].append(static_cast<int64_t>(x));
    }

    /// @brief HERE
    inline void write_add_int64(int64_t index, int64_t x) noexcept {
      buffers_int64_[(size_t)index].append(buffers_int64_[(size_t)index].last() + x);
    }

    /// @brief HERE
    inline void write_float64(int64_t index, double x) noexcept {
      buffers_float64_[(size_t)index].append(x);
    }

    /// @brief HERE
    inline int64_t get_and_increment(int64_t index) noexcept {
      return counters_[(size_t)index]++;
    }

    /// @brief HERE
    int64_t length() const noexcept {
      return length_;
    }

    /// @brief HERE
    inline void add_to_length(int64_t length) noexcept {
      length_ += length;
    }

    /// @brief HERE
    std::string debug() const noexcept;

    /// @brief HERE
    int64_t num_outputs() const {
      return (int64_t)output_names_.size();
    }

    /// @brief HERE
    std::string output_name(int64_t i) const {
      return output_names_[(size_t)i];
    }

    /// @brief HERE
    std::string output_dtype(int64_t i) const {
      switch (output_dtypes_[(size_t)i]) {
        case util::dtype::int8:
          return "int8";
        case util::dtype::uint8:
          return "uint8";
        case util::dtype::int64:
          return "int64";
        case util::dtype::float64:
          return "float64";
        default:
          return "unknown";
      }
    }

    /// @brief HERE
    int64_t output_num_items(int64_t i) const {
      switch (output_dtypes_[(size_t)i]) {
        case util::dtype::int8:
          return (int64_t)buffers_uint8_[(size_t)output_which_[(size_t)i]].nbytes();
        case util::dtype::uint8:
          return (int64_t)buffers_uint8_[(size_t)output_which_[(size_t)i]].nbytes();
        case util::dtype::int64:
          return (int64_t)buffers_int64_[(size_t)output_which_[(size_t)i]].nbytes() / 8;
        case util::dtype::float64:
          return (int64_t)buffers_float64_[(size_t)output_which_[(size_t)i]].nbytes() / 8;
        default:
          return -1;
      }
    }

    /// @brief HERE
    void output_fill(int64_t i, void* external_pointer) const {
      switch (output_dtypes_[(size_t)i]) {
        case util::dtype::int8:
          buffers_uint8_[(size_t)output_which_[(size_t)i]].concatenate(
            reinterpret_cast<uint8_t*>(external_pointer)
          );
          break;
        case util::dtype::uint8:
          buffers_uint8_[(size_t)output_which_[(size_t)i]].concatenate(
            reinterpret_cast<uint8_t*>(external_pointer)
          );
          break;
        case util::dtype::int64:
          buffers_int64_[(size_t)output_which_[(size_t)i]].concatenate(
            reinterpret_cast<int64_t*>(external_pointer)
          );
          break;
        case util::dtype::float64:
          buffers_float64_[(size_t)output_which_[(size_t)i]].concatenate(
            reinterpret_cast<double*>(external_pointer)
          );
          break;
        default:
          break;
      }
    }

  private:
    std::vector<int64_t> instructions_;
    std::vector<char> characters_;
    std::vector<int64_t> string_offsets_;

    std::vector<int64_t> record_current_field_;
    std::vector<std::vector<uint64_t>> record_checklist_init_;
    std::vector<std::vector<uint64_t>> record_checklist_;

    std::vector<std::string> output_names_;
    std::vector<util::dtype> output_dtypes_;
    std::vector<int64_t> output_which_;
    std::vector<GrowableBuffer<uint8_t>> buffers_uint8_;
    std::vector<GrowableBuffer<int64_t>> buffers_int64_;
    std::vector<GrowableBuffer<double>> buffers_float64_;

    int64_t current_instruction_;
    std::vector<int64_t> instruction_stack_;
    int64_t current_stack_depth_;
    std::vector<int64_t> counters_;

    int64_t length_;
  };

}

#endif // AWKWARD_IO_JSON_H_
