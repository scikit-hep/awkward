// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_FORTHMACHINE_H_
#define AWKWARD_FORTHMACHINE_H_

#include <set>
#include <map>

#include "awkward/common.h"
#include "awkward/util.h"
#include "awkward/forth/ForthInputBuffer.h"
#include "awkward/forth/ForthOutputBuffer.h"

namespace awkward {
  /// @class ForthMachine
  ///
  /// @brief HERE
  ///
  /// THERE
  template <typename T, typename I>
  class LIBAWKWARD_EXPORT_SYMBOL ForthMachineOf {
  public:
    ForthMachineOf(const std::string& source,
                   int64_t stack_size=1024,
                   int64_t recursion_depth=1024,
                   int64_t output_initial_size=1024,
                   double output_resize_factor=1.5);

    ~ForthMachineOf();

    /// @brief HERE
    const std::string
      source() const;

    /// @brief HERE
    const std::vector<I>
      bytecodes() const;

    /// @brief HERE
    const std::string
      assembly_instructions() const;

    /// @brief HERE
    int64_t
      stack_size() const;

    /// @brief HERE
    int64_t
      recursion_depth() const;

    /// @brief HERE
    int64_t
      output_initial_size() const;

    /// @brief HERE
    int64_t
      output_resize_factor() const;

    /// @brief HERE
    const std::vector<T>
      stack() const;

    /// @brief HERE
    const std::map<std::string, T>
      variables() const;

    /// @brief HERE
    void
      begin(const std::map<std::string, std::shared_ptr<ForthInputBuffer>>& inputs);

    /// @brief HERE
    void
      begin();

    /// @brief HERE
    const std::map<std::string, std::shared_ptr<ForthOutputBuffer>>
      step();

    /// @brief HERE
    int64_t
      current_instruction() const;

    /// @brief HERE
    void
      end();

    /// @brief HERE
    void
      run(const std::map<std::string, std::shared_ptr<ForthInputBuffer>>& inputs,
          const std::set<util::ForthError>& ignore);

    /// @brief HERE
    void
      run(const std::map<std::string, std::shared_ptr<ForthInputBuffer>>& inputs);

    /// @brief HERE
    void
      run();

    /// @brief HERE
    void
      count_reset();

    /// @brief HERE
    int64_t
      count_instructions() const;

    /// @brief HERE
    int64_t
      count_reads() const;

    /// @brief HERE
    int64_t
      count_writes() const;

    /// @brief HERE
    int64_t
      count_nanoseconds() const;

  private:
    void compile();

    std::string source_;
    int64_t output_initial_size_;
    double output_resize_factor_;

    T* stack_buffer_;
    int64_t stack_top_;
    int64_t stack_size_;

    std::vector<std::string> variable_names_;
    std::vector<T> variables_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<util::dtype> output_dtypes_;

    std::vector<int64_t> instructions_offsets_;
    std::vector<I> instructions_;

    std::vector<std::shared_ptr<ForthInputBuffer>> current_inputs_;
    std::vector<std::shared_ptr<ForthOutputBuffer>> current_outputs_;

    int64_t* current_which_;
    int64_t* current_where_;
    int64_t instruction_current_depth_;
    int64_t instruction_max_depth_;

    int64_t* do_instruction_depth_;
    int64_t* do_stop_;
    int64_t* do_i_;
    int64_t do_current_depth_;

    util::ForthError current_error_;

    int64_t count_instructions_;
    int64_t count_reads_;
    int64_t count_writes_;
    int64_t count_nanoseconds_;
  };

  using ForthMachine32 = ForthMachineOf<int32_t, int32_t>;
  using ForthMachine64 = ForthMachineOf<int64_t, int32_t>;

}

#endif // AWKWARD_FORTHMACHINE_H_
