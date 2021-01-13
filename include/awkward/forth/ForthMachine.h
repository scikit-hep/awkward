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
                   int64_t stack_max_depth=1024,
                   int64_t recursion_max_depth=1024,
                   int64_t output_initial_size=1024,
                   double output_resize_factor=1.5);

    ~ForthMachineOf();

    /// @brief HERE
    const std::string
      source() const noexcept;

    /// @brief HERE
    const std::vector<I>
      bytecodes() const noexcept;

    /// @brief HERE
    const std::vector<int64_t>
      bytecodes_offsets() const noexcept;

    /// @brief HERE
    const std::string
      assembly_instructions() const;

    /// @brief HERE
    const std::vector<std::string>
      dictionary() const;

    /// @brief HERE
    int64_t
      stack_max_depth() const noexcept;

    /// @brief HERE
    int64_t
      recursion_max_depth() const noexcept;

    /// @brief HERE
    int64_t
      output_initial_size() const noexcept;

    /// @brief HERE
    int64_t
      output_resize_factor() const noexcept;

    /// @brief HERE
    const std::vector<T>
      stack() const;

    /// @brief HERE
    T
      stack_at(int64_t from_top) const noexcept;

    /// @brief HERE
    int64_t
      stack_depth() const noexcept;

    /// @brief HERE
    bool
      stack_can_push() const noexcept;

    /// @brief HERE
    bool
      stack_can_pop() const noexcept;

    /// @brief HERE
    inline void
      stack_push(T value) noexcept;

    /// @brief HERE
    inline T
      stack_pop() noexcept;

    /// @brief HERE
    void
      stack_clear() noexcept;

    /// @brief HERE
    const std::map<std::string, T>
      variables() const;

    /// @brief HERE
    const std::vector<std::string>
      variable_index() const;

    /// @brief HERE
    T
      variable_at(const std::string& name) const;

    /// @brief HERE
    T
      variable_at(int64_t index) const noexcept;

    /// @brief HERE
    const std::map<std::string, std::shared_ptr<ForthOutputBuffer>>
      outputs() const;

    /// @brief HERE
    const std::vector<std::string>
      output_index() const noexcept;

    /// @brief HERE
    const std::shared_ptr<ForthOutputBuffer>
      output_at(const std::string& name) const;

    /// @brief HERE
    const std::shared_ptr<ForthOutputBuffer>
      output_at(int64_t index) const noexcept;

    /// @brief HERE
    void
      reset();

    /// @brief HERE
    void
      begin(const std::map<std::string, std::shared_ptr<ForthInputBuffer>>& inputs);

    /// @brief HERE
    void
      begin();

    /// @brief HERE
    util::ForthError
      step();

    /// @brief HERE
    util::ForthError
      run(const std::map<std::string, std::shared_ptr<ForthInputBuffer>>& inputs,
          const std::set<util::ForthError>& ignore);

    /// @brief HERE
    util::ForthError
      run(const std::map<std::string, std::shared_ptr<ForthInputBuffer>>& inputs);

    /// @brief HERE
    util::ForthError
      run();

    /// @brief HERE
    util::ForthError
      resume();

    /// @brief HERE
    util::ForthError
      call(const std::string& name);

    /// @brief HERE
    util::ForthError
      call(int64_t index);

    /// @brief HERE
    int64_t
      breakpoint_depth() const noexcept;

    /// @brief HERE
    int64_t
      current_bytecode() const noexcept;

    /// @brief HERE
    int64_t
      current_instruction() const noexcept;

    /// @brief HERE
    void
      count_reset() noexcept;

    /// @brief HERE
    int64_t
      count_instructions() const noexcept;

    /// @brief HERE
    int64_t
      count_reads() const noexcept;

    /// @brief HERE
    int64_t
      count_writes() const noexcept;

    /// @brief HERE
    int64_t
      count_nanoseconds() const noexcept;

  private:

    /// @brief HERE
    bool
      is_integer(const std::string& word, int64_t& value) const;

    /// @brief HERE
    bool
      is_variable(const std::string& word) const;

    /// @brief HERE
    bool
      is_input(const std::string& word) const;

    /// @brief HERE
    bool
      is_output(const std::string& word) const;

    /// @brief HERE
    bool
      is_reserved(const std::string& word) const;

    /// @brief HERE
    bool
      is_defined(const std::string& word) const;

    /// @brief HERE
    void
      compile();

    /// @brief HERE
    const std::string
      err_linecol(const std::vector<std::pair<int64_t, int64_t>>& linecol,
                  int64_t startpos,
                  int64_t stoppos,
                  const std::string& message) const;

    /// @brief HERE
    void
      parse(const std::string& defn,
            const std::vector<std::string>& tokenized,
            const std::vector<std::pair<int64_t, int64_t>>& linecol,
            int64_t start,
            int64_t stop,
            std::vector<I>& bytecodes,
            int64_t exitdepth,
            int64_t dodepth) const;

    /// @brief HERE
    inline void
      write_from_stack(int64_t num, T* top) noexcept;

    /// @brief HERE
    inline bool
      is_done() const noexcept;

    /// @brief HERE
    inline bool
      is_segment_done() const noexcept;

    /// @brief HERE
    void
      internal_run(bool keep_going); // noexcept

    /// @brief HERE
    inline T*
      stack_pop2() noexcept;

    /// @brief HERE
    inline T*
      stack_pop2_before_pushing1() noexcept;

    /// @brief HERE
    inline T*
      stack_peek() const noexcept;

    /// @brief HERE
    inline I
      instruction_get() const noexcept;

    /// @brief HERE
    inline void
      bytecodes_pointer_push(int64_t which) noexcept;

    /// @brief HERE
    inline void
      bytecodes_pointer_pop() noexcept;

    /// @brief HERE
    inline int64_t&
      bytecodes_pointer_which() const noexcept;

    /// @brief HERE
    inline int64_t&
      bytecodes_pointer_where() const noexcept;

    /// @brief HERE
    inline void
      do_loop_push(int64_t start, int64_t stop) noexcept;

    /// @brief HERE
    inline void
      do_steploop_push(int64_t start, int64_t stop) noexcept;

    /// @brief HERE
    inline int64_t&
      do_recursion_depth() const noexcept;

    /// @brief HERE
    inline int64_t
      do_abs_recursion_depth() const noexcept;

    /// @brief HERE
    inline bool
      do_loop_is_step() const noexcept;

    /// @brief HERE
    inline int64_t&
      do_stop() const noexcept;

    /// @brief HERE
    inline int64_t&
      do_i() const noexcept;

    /// @brief HERE
    inline int64_t&
      do_j() const noexcept;

    /// @brief HERE
    inline int64_t&
      do_k() const noexcept;

    std::string source_;
    int64_t output_initial_size_;
    double output_resize_factor_;

    T* stack_buffer_;
    int64_t stack_depth_;
    int64_t stack_max_depth_;

    std::vector<std::string> variable_names_;
    std::vector<T> variables_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<util::dtype> output_dtypes_;

    std::map<std::string, I> dictionary_names_;
    std::vector<std::vector<I>> dictionary_;
    std::vector<int64_t> bytecodes_offsets_;
    std::vector<I> bytecodes_;
    std::vector<int64_t> bytecode_to_instruction_;

    std::vector<std::shared_ptr<ForthInputBuffer>> current_inputs_;
    std::vector<std::shared_ptr<ForthOutputBuffer>> current_outputs_;

    int64_t current_breakpoint_depth_;

    int64_t* current_which_;
    int64_t* current_where_;
    int64_t recursion_current_depth_;
    int64_t recursion_max_depth_;

    int64_t* do_recursion_depth_;
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
