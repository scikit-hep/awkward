// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#include "awkward/forth/ForthMachine.h"

namespace awkward {
  template <typename T, typename I>
  ForthMachineOf<T, I>::ForthMachineOf(const std::string& source,
                                       int64_t stack_size,
                                       int64_t recursion_depth,
                                       int64_t output_initial_size,
                                       double output_resize_factor)
    : source_(source)
    , output_initial_size_(output_initial_size)
    , output_resize_factor_(output_resize_factor)

    , stack_buffer_(new T[stack_size])
    , stack_top_(0)
    , stack_size_(stack_size)

    , current_inputs_()
    , current_outputs_()

    , current_which_(new int64_t[recursion_depth])
    , current_where_(new int64_t[recursion_depth])
    , instruction_current_depth_(0)
    , instruction_max_depth_(recursion_depth)

    , do_instruction_depth_(new int64_t[recursion_depth])
    , do_stop_(new int64_t[recursion_depth])
    , do_i_(new int64_t[recursion_depth])
    , do_current_depth_(0)

    , current_error_(util::ForthError::none)

    , count_instructions_(0)
    , count_reads_(0)
    , count_writes_(0)
    , count_nanoseconds_(0)
  {
    compile();
  }

  template <typename T, typename I>
  void ForthMachineOf<T, I>::compile() {

  }

  template class EXPORT_TEMPLATE_INST ForthMachineOf<int32_t, int32_t>;
  template class EXPORT_TEMPLATE_INST ForthMachineOf<int64_t, int32_t>;

}
