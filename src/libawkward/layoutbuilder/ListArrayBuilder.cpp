// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/layoutbuilder/ListArrayBuilder.cpp", line)

#include "awkward/layoutbuilder/ListArrayBuilder.h"
#include "awkward/layoutbuilder/LayoutBuilder.h"

namespace awkward {

  ///
  template <typename T, typename I>
  ListArrayBuilder<T, I>::ListArrayBuilder(FormBuilderPtr<T, I> content,
                                     const util::Parameters& parameters,
                                     const std::string& form_key,
                                     const std::string& form_starts,
                                     const std::string attribute,
                                     const std::string partition)
    : content_(content),
      parameters_(parameters),
      begun_(false),
      form_starts_(form_starts),
      form_key_(form_key) {
    vm_output_data_ = std::string("part")
      .append(partition).append("-")
      .append(form_key).append("-")
      .append(attribute);

    vm_func_name_ = std::string(form_key).append("-").append(attribute);

    vm_output_ = std::string("output ")
      .append(vm_output_data_)
      .append(" ")
      .append(form_starts)
      .append(" ")
      .append(content_.get()->vm_output());

    vm_func_.append(content_.get()->vm_func())
      .append(": ").append(vm_func_name()).append(" ")
      .append(std::to_string(static_cast<utype>(state::begin_list)))
      .append(" <> if ")
      .append(std::to_string(LayoutBuilder<T, I>::next_error_id())).append(" err ! err @ halt ")
      .append("then 0 begin pause dup ")
      .append(std::to_string(static_cast<utype>(state::end_list)))
      .append(" = if drop ")
      .append(vm_output_data_).append(" +<- stack exit else ")
      .append(content_.get()->vm_func_name()).append(" ")
      .append("1+ then again ; ");

    vm_data_from_stack_ = std::string(content_.get()->vm_from_stack())
      .append("0 ").append(vm_output_data_).append(" <- stack ");

    vm_error_.append(content_.get()->vm_error());
    vm_error_.append("s\"ListArray Builder needs begin_list\" ");
  }

  template <typename T, typename I>
  const std::string
  ListArrayBuilder<T, I>::classname() const {
    return "ListArrayBuilder";
  }

  template <typename T, typename I>
  const std::string
  ListArrayBuilder<T, I>::to_buffers(
    BuffersContainer& container,
    const ForthOutputBufferMap& outputs) const {
    auto search = outputs.find(vm_output_data());
    if (search != outputs.end()) {
      auto offsets = search->second.get()->toIndex64();

      // FIXME: deal with complex numbers in the builder itself
      if (content().get()->is_complex()) {
        for (int64_t i = 0; i < offsets.length(); i++) {
          offsets.ptr().get()[i] = offsets.ptr().get()[i] >> 1;
        }
      }

      container.copy_buffer(form_key() + "-offsets",
                            offsets.ptr().get(),
                            (int64_t)(offsets.length() * (int64_t)sizeof(int64_t)));

      return "{\"class\": \"ListOffsetArray\", \"offsets\": \"i64\", \"content\": "
        + content()->to_buffers(container, outputs) + ", "
        + this->parameters_as_string(parameters_) + " \"form_key\": \""
        + form_key() + "\"}";
    }
    throw std::invalid_argument(
      std::string("Snapshot of a ") + classname()
      + std::string(" needs offsets ")
      + FILENAME(__LINE__));
  }

  template <typename T, typename I>
  const std::string
  ListArrayBuilder<T, I>::vm_output() const {
    return vm_output_;
  }

  template <typename T, typename I>
  const std::string
  ListArrayBuilder<T, I>::vm_output_data() const {
    return vm_output_data_;
  }

  template <typename T, typename I>
  const std::string
  ListArrayBuilder<T, I>::vm_func() const {
    return vm_func_;
  }

  template <typename T, typename I>
  const std::string
  ListArrayBuilder<T, I>::vm_func_name() const {
    return vm_func_name_;
  }

  template <typename T, typename I>
  const std::string
  ListArrayBuilder<T, I>::vm_func_type() const {
    return vm_func_type_;
  }

  template <typename T, typename I>
  const std::string
  ListArrayBuilder<T, I>::vm_from_stack() const {
    return vm_data_from_stack_;
  }

  template <typename T, typename I>
  const std::string
  ListArrayBuilder<T, I>::vm_error() const {
    return vm_error_;
  }

  template <typename T, typename I>
  void
  ListArrayBuilder<T, I>::boolean(bool x, LayoutBuilderPtr<T, I> builder) {
    content_.get()->boolean(x, builder);
  }

  template <typename T, typename I>
  void
  ListArrayBuilder<T, I>::int64(int64_t x, LayoutBuilderPtr<T, I> builder) {
    content_.get()->int64(x, builder);
  }

  template <typename T, typename I>
  void
  ListArrayBuilder<T, I>::float64(double x, LayoutBuilderPtr<T, I> builder) {
    content_.get()->float64(x, builder);
  }

  template <typename T, typename I>
  void
  ListArrayBuilder<T, I>::complex(std::complex<double> x, LayoutBuilderPtr<T, I> builder) {
    content_.get()->complex(x, builder);
  }

  template <typename T, typename I>
  void
  ListArrayBuilder<T, I>::bytestring(const std::string& x, LayoutBuilderPtr<T, I> builder) {
    content_.get()->bytestring(x, builder);
  }

  template <typename T, typename I>
  void
  ListArrayBuilder<T, I>::string(const std::string& x, LayoutBuilderPtr<T, I> builder) {
    content_.get()->string(x, builder);
  }

  template <typename T, typename I>
  void
  ListArrayBuilder<T, I>::begin_list(LayoutBuilderPtr<T, I> builder) {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'end_list' without 'begin_list' at the same level before it")
        + FILENAME(__LINE__));
    }
    else if (!content_.get()->active()) {
      builder->add_end_list();
      begun_ = false;
    }
    else {
      content_.get()->end_list(builder);
    }
  }

  template <typename T, typename I>
  void
  ListArrayBuilder<T, I>::end_list(LayoutBuilderPtr<T, I> builder) {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'end_list' without 'begin_list' at the same level before it")
        + FILENAME(__LINE__));
    }
    else if (!content_.get()->active()) {
      builder->add_end_list();
      begun_ = false;
    }
    else {
      content_.get()->end_list(builder);
    }
  }

  template <typename T, typename I>
  bool
  ListArrayBuilder<T, I>::active() {
    return begun_;
  }

  template class EXPORT_TEMPLATE_INST ListArrayBuilder<int32_t, int32_t>;
  template class EXPORT_TEMPLATE_INST ListArrayBuilder<int64_t, int32_t>;

}
