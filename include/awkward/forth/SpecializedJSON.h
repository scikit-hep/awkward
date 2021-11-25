// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_SPECIALIZEDJSON_H_
#define AWKWARD_SPECIALIZEDJSON_H_

#include "awkward/common.h"
#include "awkward/util.h"
#include "awkward/forth/ForthOutputBuffer.h"

namespace awkward {
  /// @class SpecializedJSON
  ///
  /// @brief HERE
  ///
  /// THERE
  class LIBAWKWARD_EXPORT_SYMBOL SpecializedJSON {
  public:
    SpecializedJSON(const std::string& jsonassembly,
                    int64_t output_initial_size,
                    double output_resize_factor);

    /// @brief HERE
    const std::shared_ptr<ForthOutputBuffer>
      output_at(const std::string& name) const;

    /// @brief HERE
    util::dtype
      dtype_at(const std::string& name) const;

    /// @brief HERE
    int64_t
      length() const noexcept;

    /// @brief HERE
    int64_t
      json_position() const noexcept;

    /// @brief HERE
    bool parse_string(const char* source) noexcept;

    /// @brief HERE
    void reset() noexcept;

    /// @brief HERE
    std::string debug() const noexcept;

    /// @brief HERE
    std::string debug_listing() const noexcept;

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
        if (strncmp(str, &chars[start], stop - start) == 0) {
          return i - stringsstart;
        }
      }
      return -1;
    }

    /// @brief HERE
    inline int64_t find_key(const char* str) noexcept {
      int64_t* offsets = string_offsets_.data();
      char* chars = characters_.data();
      int64_t stringi;
      int64_t start;
      int64_t stop;
      for (int64_t i = current_instruction_ + 1;  i <= current_instruction_ + argument1();  i++) {
        stringi = instructions_.data()[i * 4 + 1];
        start = offsets[stringi];
        stop = offsets[stringi + 1];
        if (strncmp(str, &chars[start], stop - start) == 0) {
          return instructions_.data()[i * 4 + 2];
        }
      }
      return -1;
    }

    /// @brief HERE
    inline void write_int8(int64_t index, int8_t x) noexcept {
      outputs_[index].get()->write_one_int8(x, false);
    }

    /// @brief HERE
    inline void write_int64(int64_t index, int64_t x) noexcept {
      outputs_[index].get()->write_one_int64(x, false);
    }

    /// @brief HERE
    inline void write_uint64(int64_t index, int64_t x) noexcept {
      outputs_[index].get()->write_one_uint64(x, false);
    }

    /// @brief HERE
    inline void write_many_uint8(int64_t index, int64_t num_items, const uint8_t* values) noexcept {
      outputs_[index].get()->write_const_uint8(num_items, values);
    }

    /// @brief HERE
    inline void write_add_int64(int64_t index, int64_t x) noexcept {
      outputs_[index].get()->write_add_int64(x);
    }

    /// @brief HERE
    inline void write_float64(int64_t index, double x) noexcept {
      outputs_[index].get()->write_one_float64(x, false);
    }

    /// @brief HERE
    inline int64_t get_and_increment(int64_t index) noexcept {
      return counters_[index]++;
    }

    /// @brief HERE
    inline void set_length(int64_t length) noexcept {
      length_ = length;
    }

  private:
    int64_t output_index(const std::string& name,
                         util::dtype dtype,
                         bool leading_zero,
                         int64_t init,
                         double resize);

    std::vector<std::string> output_names_;
    std::vector<util::dtype> output_dtypes_;
    std::vector<std::shared_ptr<ForthOutputBuffer>> outputs_;
    std::vector<bool> output_leading_zero_;

    std::vector<int64_t> instructions_;
    std::vector<char> characters_;
    std::vector<int64_t> string_offsets_;

    int64_t current_instruction_;
    std::vector<int64_t> instruction_stack_;
    int64_t current_stack_depth_;
    std::vector<int64_t> counters_;

    int64_t length_;

    int64_t json_position_;
  };
}

#endif // AWKWARD_SPECIALIZEDJSON_H_
