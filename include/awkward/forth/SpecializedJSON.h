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
    bool parse_string(const char* source) noexcept;

    /// @brief HERE
    void reset() noexcept;

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
    inline void push_instruction_stack(int64_t jump_to) noexcept {
      instruction_stack_[current_stack_depth_] = current_instruction_;
      current_stack_depth_++;
      current_instruction_ = jump_to;
    }

    /// @brief HERE
    inline void pop_instruction_stack() noexcept {
      current_stack_depth_--;
      current_instruction_ = instruction_stack_[current_stack_depth_];
    }

    /// @brief HERE
    inline void write_int64(int64_t index, int64_t x) noexcept {
      outputs_[index].get()->write_one_int64(x, false);
    }

    /// @brief HERE
    inline void write_add_int64(int64_t index, int64_t x) noexcept {
      outputs_[index].get()->write_add_int64(x);
    }

    /// @brief HERE
    inline void write_float64(int64_t index, double x) noexcept {
      outputs_[index].get()->write_one_float64(x, false);
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
    std::vector<uint8_t> characters_;
    std::vector<int64_t> string_offsets_;

    int64_t current_instruction_;
    std::vector<int64_t> instruction_stack_;
    int64_t current_stack_depth_;
    std::vector<int64_t> counters_;

    int64_t json_position_;
  };
}

#endif // AWKWARD_SPECIALIZEDJSON_H_
