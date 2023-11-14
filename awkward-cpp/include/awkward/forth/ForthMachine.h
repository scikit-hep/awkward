// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#ifndef AWKWARD_FORTHMACHINE_H_
#define AWKWARD_FORTHMACHINE_H_

#include <set>
#include <map>
#include <stack>

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
  class EXPORT_SYMBOL ForthMachineOf {

    template <typename TYPE> using IndexTypeOf = typename std::vector<TYPE>::size_type;

  public:
    ForthMachineOf(const std::string& source,
                   int64_t stack_max_depth=1024,
                   int64_t recursion_max_depth=1024,
                   int64_t string_buffer_size=1024,
                   int64_t output_initial_size=1024,
                   double output_resize_factor=1.5);

    ~ForthMachineOf();

    /// @brief HERE
    int64_t
      abi_version() const noexcept;

    /// @brief HERE
    const std::string
      source() const noexcept;

    /// @brief HERE
    const std::vector<I>
      bytecodes() const;

    /// @brief HERE
    const std::vector<int64_t>
      bytecodes_offsets() const;

    /// @brief HERE
    const std::string
      decompiled() const;

    /// @brief HERE
    const std::string
      decompiled_segment(int64_t segment_position, const std::string& indent="",
                         bool endline = true) const;

    /// @brief HERE
    const std::string
      decompiled_at(int64_t bytecode_position, const std::string& indent="") const;

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
      string_buffer_size() const noexcept;

    /// @brief HERE
    int64_t
      output_initial_size() const noexcept;

    /// @brief HERE
    double
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
    inline bool
      stack_can_push() const noexcept {
      return stack_depth_ < stack_max_depth_;
    }

    /// @brief HERE
    inline bool
      stack_can_pop() const noexcept {
      return stack_depth_ > 0;
    }

    /// @brief HERE
    inline void
      stack_push(T value) noexcept {
      stack_buffer_[stack_depth_] = value;
      stack_depth_++;
    }

    /// @brief HERE
    inline T
      stack_pop() noexcept {
      stack_depth_--;
      return stack_buffer_[stack_depth_];
    }

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
    bool
      input_must_be_writable(const std::string& name) const;

    /// @brief HERE
    int64_t
      input_position_at(const std::string& name) const;

    /// @brief HERE
    int64_t
      input_position_at(int64_t index) const noexcept;

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

    /// @brief Returns a string at 'index'.
    /// The strings are defined with an 's"' core word.
    const std::string
      string_at(int64_t index) const noexcept;

    /// @brief HERE
    void
      reset();

    /// @brief HERE
    void
      begin(const std::map<std::string, std::shared_ptr<ForthInputBuffer>>& inputs);

    /// @brief HERE
    void
      begin();

    ///@brief HERE
    util::ForthError
    begin_again(const std::map<std::string, std::shared_ptr<ForthInputBuffer>>& inputs, bool reset_instruction);

    /// @brief HERE
    util::ForthError
      step();

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
    void
      maybe_throw(util::ForthError err, const std::set<util::ForthError>& ignore) const;

    /// @brief HERE
    int64_t
      current_bytecode_position() const noexcept;

    /// @brief HERE
    int64_t
      current_recursion_depth() const noexcept;

    /// @brief HERE
    const std::string
      current_instruction() const;

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
      is_nbit(const std::string& word, I& value) const;

    /// @brief HERE
    bool
      is_reserved(const std::string& word) const;

    /// @brief HERE
    bool
      is_defined(const std::string& word) const;

    /// @brief HERE
    inline bool
      is_ready() const noexcept {
      return is_ready_;
    }

    /// @brief HERE
    inline bool
      is_done() const noexcept {
      return recursion_target_depth_.empty();
    }

    /// @brief HERE
    inline bool
      is_segment_done() const noexcept {
      return !(bytecodes_pointer_where() < (
                   bytecodes_offsets_[(IndexTypeOf<int64_t>)bytecodes_pointer_which() + 1] -
                   bytecodes_offsets_[(IndexTypeOf<int64_t>)bytecodes_pointer_which()]
               ));
    }

  private:
    /// @brief HERE
    bool
    segment_nonempty(int64_t segment_position) const;

    /// @brief HERE
    int64_t
    bytecodes_per_instruction(int64_t bytecode_position) const;

    /// @brief HERE
    const std::string
      err_linecol(const std::vector<std::pair<int64_t, int64_t>>& linecol,
                  int64_t startpos,
                  int64_t stoppos,
                  const std::string& message) const;

    /// @brief HERE
    void
      tokenize(std::vector<std::string>& tokenized,
               std::vector<std::pair<int64_t, int64_t>>& linecol);

    /// @brief HERE
    void
      compile(const std::vector<std::string>& tokenized,
              const std::vector<std::pair<int64_t, int64_t>>& linecol);

    /// @brief HERE
    void
      parse(const std::string& defn,
            const std::vector<std::string>& tokenized,
            const std::vector<std::pair<int64_t, int64_t>>& linecol,
            int64_t start,
            int64_t stop,
            std::vector<I>& bytecodes,
            std::vector<std::vector<I>>& dictionary,
            int64_t exitdepth,
            int64_t dodepth);

    /// @brief HERE
    void
      internal_run(bool single_step, int64_t recursion_target_depth_top); // noexcept

    /// @brief HERE
    void
      write_from_stack(int64_t num, T* top) noexcept;

    /// @brief HERE
    void
      write_add_from_stack(int64_t num, T* top) noexcept;

    /// @brief HERE
    void
      print_number(T num) noexcept;

    /// @brief HERE
    inline bool
      stack_cannot_push() const noexcept {
      return stack_depth_ == stack_max_depth_;
    }

    /// @brief HERE
    inline bool
      stack_cannot_pop() const noexcept {
      return stack_depth_ == 0;
    }

    /// @brief HERE
    inline bool
      stack_cannot_pop2() const noexcept {
      return stack_depth_ < 2;
    }

    /// @brief HERE
    inline bool
      stack_cannot_pop3() const noexcept {
      return stack_depth_ < 3;
    }

    /// @brief HERE
    inline T*
      stack_pop2() noexcept {
      stack_depth_ -= 2;
      return &stack_buffer_[stack_depth_];
    }

    /// @brief HERE
    inline T*
      stack_pop2_before_pushing1() noexcept {
      stack_depth_--;
      return &stack_buffer_[stack_depth_ - 1];
    }

    /// @brief HERE
    inline T*
      stack_peek() const noexcept {
      return &stack_buffer_[stack_depth_ - 1];
    }

    /// @brief HERE
    inline I
      bytecode_get() const noexcept {
      int64_t start = bytecodes_offsets_[(IndexTypeOf<int64_t>)bytecodes_pointer_which()];
      return bytecodes_[(IndexTypeOf<I>)(start + bytecodes_pointer_where())];
    }

    /// @brief HERE
    inline void
      bytecodes_pointer_push(int64_t which) noexcept {
      current_which_[recursion_current_depth_] = which;
      current_where_[recursion_current_depth_] = 0;
      recursion_current_depth_++;
    }

    /// @brief HERE
    inline void
      bytecodes_pointer_pop() noexcept {
      recursion_current_depth_--;
    }

    /// @brief HERE
    inline int64_t&
      bytecodes_pointer_which() const noexcept {
      return current_which_[recursion_current_depth_ - 1];
    }

    /// @brief HERE
    inline int64_t&
      bytecodes_pointer_where() const noexcept {
      return current_where_[recursion_current_depth_ - 1];
    }

    /// @brief HERE
    inline void
      do_loop_push(int64_t start, int64_t stop) noexcept {
      do_recursion_depth_[do_current_depth_] = recursion_current_depth_;
      do_stop_[do_current_depth_] = stop;
      do_i_[do_current_depth_] = start;
      do_current_depth_++;
    }

    /// @brief HERE
    inline void
      do_steploop_push(int64_t start, int64_t stop) noexcept {
      do_recursion_depth_[do_current_depth_] = ~recursion_current_depth_;
      do_stop_[do_current_depth_] = stop;
      do_i_[do_current_depth_] = start;
      do_current_depth_++;
    }

    /// @brief HERE
    inline int64_t&
      do_recursion_depth() const noexcept {
      return do_recursion_depth_[do_current_depth_ - 1];
    }

    /// @brief HERE
    inline int64_t
    do_abs_recursion_depth() const noexcept {
      int64_t out = do_recursion_depth_[do_current_depth_ - 1];
      if (out < 0) {
        return ~out;
      }
      else {
        return out;
      }
    }

    /// @brief HERE
    inline bool
      do_loop_is_step() const noexcept {
      return do_recursion_depth_[do_current_depth_ - 1] < 0;
    }

    /// @brief HERE
    inline int64_t&
      do_stop() const noexcept {
      return do_stop_[do_current_depth_ - 1];
    }

    /// @brief HERE
    inline int64_t&
      do_i() const noexcept {
      return do_i_[do_current_depth_ - 1];
    }

    /// @brief HERE
    inline int64_t&
      do_j() const noexcept {
      return do_i_[do_current_depth_ - 2];
    }

    /// @brief HERE
    inline int64_t&
      do_k() const noexcept {
      return do_i_[do_current_depth_ - 3];
    }

    std::string source_;
    int64_t output_initial_size_;
    double output_resize_factor_;

    T* stack_buffer_;
    int64_t stack_depth_;
    int64_t stack_max_depth_;

    std::vector<std::string> variable_names_;
    std::vector<T> variables_;

    std::vector<std::string> input_names_;
    std::vector<bool> input_must_be_writable_;
    std::vector<std::string> output_names_;
    std::vector<util::dtype> output_dtypes_;

    std::vector<std::string> strings_;
    std::vector<std::string> dictionary_names_;
    std::vector<I> dictionary_bytecodes_;
    std::vector<int64_t> bytecodes_offsets_;
    std::vector<I> bytecodes_;

    char* string_buffer_;
    int64_t string_buffer_size_;

    std::vector<std::shared_ptr<ForthInputBuffer>> current_inputs_;
    std::vector<std::shared_ptr<ForthOutputBuffer>> current_outputs_;
    bool is_ready_;

    int64_t* current_which_;
    int64_t* current_where_;
    int64_t recursion_current_depth_;
    std::stack<int64_t> recursion_target_depth_;
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
